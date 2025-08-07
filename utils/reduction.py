# Copied from https://github.com/pytorch/pytorch/blob/main/torch/multiprocessing/reductions.py
# Replaced multiprocessing -> multiprocess

# mypy: allow-untyped-defs
import multiprocess as multiprocessing
import os
import threading
from multiprocess import reduction
from multiprocess.util import register_after_fork
from typing import Union

import torch
from torch._namedtensor_internals import check_serializing_named_tensor


try:
    # Early load resource_sharer to prevent a partially initialized instance
    # from being inherited in a forked child process. The reduce_storage method
    # requires this module indirectly through DupFd(). The built-in mp.Queue
    # class pickles arguments in a background thread which may overlap with the
    # fork.
    import multiprocess.resource_sharer
except ImportError:
    pass


class StorageWeakRef:
    r"""A weak reference to a Storage.

    The cdata member is a Python number containing the integer representation of
    the Storage pointer.
    """

    __slots__ = ["cdata", "_free_weak_ref"]

    def __init__(self, storage):
        self.cdata = storage._weak_ref()
        # Save a direct reference to _free_weak_ref because the `torch` module
        # might be cleared during Python shutdown before this module is cleared.
        self._free_weak_ref = torch.Storage._free_weak_ref  # type: ignore[attr-defined]

    @classmethod
    def from_weakref(cls, cdata):
        instance = cls.__new__(cls)
        instance.cdata = cdata
        instance._free_weak_ref = torch.Storage._free_weak_ref  # type: ignore[attr-defined]
        return instance

    def expired(self):
        return torch.Storage._expired(self.cdata)  # type: ignore[attr-defined]

    def __del__(self):
        self._free_weak_ref(self.cdata)

    def __hash__(self):
        return self.cdata

    def __eq__(self, other):
        if id(self) == id(other):
            return True
        return self.cdata == other.cdata


class SharedCache(dict):
    """Dictionary from multiprocessing handles to StorageWeakRef."""

    def __init__(self) -> None:
        # free_dead_references() is called if the len exceeds the current
        # limit. The limit scales with the number of remaining live objects.
        self.limit = 128
        # `fork` inherits lock state, so in case we fork when the lock is held,
        # we register a function to reset the lock to a new object to avoid
        # possible deadlocks, following python multiprocessing library design.
        self._after_fork()
        register_after_fork(self, SharedCache._after_fork)

    def _after_fork(self):
        self.lock = threading.Lock()

    def get(self, key):  # type: ignore[override]
        with self.lock:
            return dict.get(self, key)

    def __setitem__(self, key, storage_ref):
        with self.lock:
            dict.__setitem__(self, key, storage_ref)
            if len(self) > self.limit:
                self.free_dead_references()

    def free_dead_references(self):
        live = 0
        for key, storage_ref in list(self.items()):
            if storage_ref.expired():
                del self[key]
            else:
                live += 1
        self.limit = max(128, live * 2)


# mapping from handles to StorageWeakRef objects
_shared_cache = SharedCache()


def rebuild_event(device, handle):
    return torch.cuda.Event(handle=handle, device=device)


def reduce_event(event):
    return rebuild_event, (event.device, event.ipc_handle)


def rebuild_tensor(cls, storage, metadata):
    storage_offset, size, stride, requires_grad, backward_hooks, metadata = metadata
    t = cls._new_with_storage(storage, storage_offset, size, stride)
    if cls in (torch.HalfTensor, torch.BFloat16Tensor):
        # For half and bfloat16 tensors, we need to check if the storage is
        # actually the right type. This is because when we serialize a half
        # tensor, we convert it to float16, but when we deserialize it, we
        # want to keep it as half.
        if t.dtype != cls._scalar_type:
            t = t.to(cls._scalar_type)
    t.requires_grad = requires_grad
    if backward_hooks:
        t._backward_hooks = backward_hooks

    # NB: This line exists only to backwards compatibility with the
    # PyTorch 1.2 serialization format.  If you're running this code
    # on a serialized tensor from 1.2, this will cause the tensor to
    # be permuted to the old 1.2 format.  The 1.2 format is different
    # from the 1.3+ format in that 1.2 uses the legacy TH storage
    # accessor, while 1.3+ uses the c10 storage accessor.  The
    # storage accessor is what determines the row-major vs column-major
    # orientation of the tensor.  Since the legacy TH accessor is
    # row-major, and the c10 accessor is column-major, we need to
    # permute the tensor to maintain compatibility.
    #
    # The permute is done as follows:
    # - If the tensor is 1D, do nothing
    # - If the tensor is 2D, transpose it
    # - If the tensor is 3D, permute it to (2, 0, 1)
    # - If the tensor is 4D, permute it to (3, 2, 0, 1)
    # - And so on
    if metadata is not None:
        t = t.permute(*metadata)

    return t


def rebuild_meta_tensor(
    tensor_cls,
    tensor_size,
    tensor_stride,
    tensor_offset,
    dtype,
    storage_size_bytes,
    requires_grad,
):
    storage = tensor_cls._new_with_storage(
        torch.UntypedStorage._new_with_file(
            None, storage_size_bytes, dtype, tensor_cls._element_size()
        ),
        tensor_offset,
        tensor_size,
        tensor_stride,
    )
    storage.requires_grad = requires_grad
    return storage


def rebuild_cuda_tensor(
    tensor_cls,
    tensor_size,
    tensor_stride,
    tensor_offset,
    storage_cls,
    dtype,
    storage_device,
    storage_handle,
    storage_size_bytes,
    storage_offset_bytes,
    requires_grad,
    ref_counter_handle,
    ref_counter_offset,
    event_handle,
    event_sync_required,
):
    # If storage_handle is None, storage points to nullptr.
    if storage_handle is None:
        storage = storage_cls(0, dtype=dtype, device=storage_device)
    else:
        storage = storage_cls._new_with_cuda_file(
            storage_handle,
            storage_size_bytes,
            dtype,
            storage_device,
            storage_offset_bytes,
        )

    # NB: This line exists only to backwards compatibility with the
    # PyTorch 1.2 serialization format.  If you're running this code
    # on a serialized tensor from 1.2, this will cause the tensor to
    # be permuted to the old 1.2 format.  The 1.2 format is different
    # from the 1.3+ format in that 1.2 uses the legacy TH storage
    # accessor, while 1.3+ uses the c10 storage accessor.  The
    # storage accessor is what determines the row-major vs column-major
    # orientation of the tensor.  Since the legacy TH accessor is
    # row-major, and the c10 accessor is column-major, we need to
    # permute the tensor to maintain compatibility.
    #
    # The permute is done as follows:
    # - If the tensor is 1D, do nothing
    # - If the tensor is 2D, transpose it
    # - If the tensor is 3D, permute it to (2, 0, 1)
    # - If the tensor is 4D, permute it to (3, 2, 0, 1)
    # - And so on
    if tensor_size:
        t = tensor_cls._new_with_storage(
            storage, tensor_offset, tensor_size, tensor_stride
        )
    else:
        t = tensor_cls._new_with_storage(storage, tensor_offset, tensor_size, tensor_stride)

    t.requires_grad = requires_grad

    # If the tensor has a ref counter, we need to set it up.
    if ref_counter_handle is not None:
        t._set_ref_counter(
            torch.cuda._UntypedStorage._new_with_cuda_file(
                ref_counter_handle, 8, torch.int64, storage_device, ref_counter_offset
            )
        )

    # If the tensor has an event, we need to set it up.
    if event_handle is not None:
        t._set_event(
            torch.cuda.Event(handle=event_handle, device=storage_device),
            event_sync_required,
        )

    return t


def reduce_tensor(tensor):
    if tensor.requires_grad and not tensor.is_leaf:
        raise RuntimeError(
            "Cowardly refusing to serialize non-leaf tensor that requires_grad, "
            "since this will likely cause the gradient to be incorrect. "
            "If you think this is a false positive, please file an issue "
            "at https://github.com/pytorch/pytorch/issues/new?template=bug-report.md"
        )

    check_serializing_named_tensor(tensor)

    # Note: _write_file is implemented by the concrete derived class
    # (e.g., torch.HalfTensor, torch.FloatTensor, etc.)
    # to dispatch to the correct storage type
    storage = tensor.storage()

    if storage.is_cuda:
        # This is a CUDA tensor
        storage_handle = storage._share_cuda_()
        storage_size_bytes = storage.size() * storage.element_size()
        storage_offset_bytes = tensor.storage_offset() * storage.element_size()
        ref_counter_handle = None
        ref_counter_offset = 0
        event_handle = None
        event_sync_required = False

        # If the tensor has a ref counter, we need to share it.
        if hasattr(tensor, "_cdata") and tensor._cdata.ref_counter is not None:
            ref_counter = tensor._cdata.ref_counter
            ref_counter_handle = ref_counter._share_cuda_()
            ref_counter_offset = ref_counter.storage_offset() * ref_counter.element_size()

        # If the tensor has an event, we need to share it.
        if hasattr(tensor, "_cdata") and tensor._cdata.event is not None:
            event = tensor._cdata.event
            event_handle = event.ipc_handle
            event_sync_required = event.recorded

        return (
            rebuild_cuda_tensor,
            (
                type(tensor),
                tensor.size(),
                tensor.stride(),
                tensor.storage_offset(),
                type(storage),
                tensor.dtype,
                storage.device,
                storage_handle,
                storage_size_bytes,
                storage_offset_bytes,
                tensor.requires_grad,
                ref_counter_handle,
                ref_counter_offset,
                event_handle,
                event_sync_required,
            ),
        )
    else:
        # This is a CPU tensor
        if storage.is_shared:
            # This is a shared memory tensor
            storage_handle = storage._share_filename_()
            storage_size_bytes = storage.size() * storage.element_size()
            storage_offset_bytes = tensor.storage_offset() * storage.element_size()
            return (
                rebuild_cuda_tensor,
                (
                    type(tensor),
                    tensor.size(),
                    tensor.stride(),
                    tensor.storage_offset(),
                    type(storage),
                    tensor.dtype,
                    storage.device,
                    storage_handle,
                    storage_size_bytes,
                    storage_offset_bytes,
                    tensor.requires_grad,
                    None,
                    0,
                    None,
                    False,
                ),
            )
        else:
            # This is a regular CPU tensor
            storage_handle = storage._share_filename_()
            storage_size_bytes = storage.size() * storage.element_size()
            storage_offset_bytes = tensor.storage_offset() * storage.element_size()
            return (
                rebuild_cuda_tensor,
                (
                    type(tensor),
                    tensor.size(),
                    tensor.stride(),
                    tensor.storage_offset(),
                    type(storage),
                    tensor.dtype,
                    storage.device,
                    storage_handle,
                    storage_size_bytes,
                    storage_offset_bytes,
                    tensor.requires_grad,
                    None,
                    0,
                    None,
                    False,
                ),
            )


def rebuild_nested_tensor(
    rebuild_buffer_func,
    rebuild_buffer_args,
    rebuild_sizes_func,
    rebuild_sizes_args,
    rebuild_strides_func,
    rebuild_strides_args,
    rebuild_offsets_func,
    rebuild_offsets_args,
):
    buffer = rebuild_buffer_func(*rebuild_buffer_args)
    sizes = rebuild_sizes_func(*rebuild_sizes_args)
    strides = rebuild_strides_func(*rebuild_strides_args)
    offsets = rebuild_offsets_func(*rebuild_offsets_args)
    return torch._nested_tensor_from_tensor_list(buffer, sizes, strides, offsets)


def reduce_nested_tensor(nt):
    buffer = nt._buffer
    sizes = nt._nested_tensor_size()
    strides = nt._nested_tensor_strides()
    offsets = nt._nested_tensor_storage_offsets()

    # Rebuild functions for each component
    rebuild_buffer_func, rebuild_buffer_args = reduce_tensor(buffer)
    rebuild_sizes_func, rebuild_sizes_args = reduce_tensor(sizes)
    rebuild_strides_func, rebuild_strides_args = reduce_tensor(strides)
    rebuild_offsets_func, rebuild_offsets_args = reduce_tensor(offsets)

    return (
        rebuild_nested_tensor,
        (
            rebuild_buffer_func,
            rebuild_buffer_args,
            rebuild_sizes_func,
            rebuild_sizes_args,
            rebuild_strides_func,
            rebuild_strides_args,
            rebuild_offsets_func,
            rebuild_offsets_args,
        ),
    )


def rebuild_sparse_coo_tensor(
    rebuild_indices_func,
    rebuild_indices_args,
    rebuild_values_func,
    rebuild_values_args,
    shape,
    is_coalesced,
):
    indices = rebuild_indices_func(*rebuild_indices_args)
    values = rebuild_values_func(*rebuild_values_args)
    return torch.sparse_coo_tensor(indices, values, shape, is_coalesced=is_coalesced)


def rebuild_sparse_compressed_tensor(
    rebuild_compressed_indices_func,
    rebuild_compressed_indices_args,
    rebuild_plain_indices_func,
    rebuild_plain_indices_args,
    rebuild_values_func,
    rebuild_values_args,
    shape,
    layout,
):
    compressed_indices = rebuild_compressed_indices_func(*rebuild_compressed_indices_args)
    plain_indices = rebuild_plain_indices_func(*rebuild_plain_indices_args)
    values = rebuild_values_func(*rebuild_values_args)
    return torch.sparse_compressed_tensor(
        compressed_indices, plain_indices, values, shape, layout=layout
    )


def reduce_sparse_tensor(sparse):
    if sparse.layout == torch.sparse_coo:
        indices = sparse._indices()
        values = sparse._values()
        shape = sparse.size()
        is_coalesced = sparse.is_coalesced()

        rebuild_indices_func, rebuild_indices_args = reduce_tensor(indices)
        rebuild_values_func, rebuild_values_args = reduce_tensor(values)

        return (
            rebuild_sparse_coo_tensor,
            (
                rebuild_indices_func,
                rebuild_indices_args,
                rebuild_values_func,
                rebuild_values_args,
                shape,
                is_coalesced,
            ),
        )
    elif sparse.layout in (torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc):
        compressed_indices = sparse._compressed_indices()
        plain_indices = sparse._plain_indices()
        values = sparse._values()
        shape = sparse.size()
        layout = sparse.layout

        rebuild_compressed_indices_func, rebuild_compressed_indices_args = reduce_tensor(
            compressed_indices
        )
        rebuild_plain_indices_func, rebuild_plain_indices_args = reduce_tensor(plain_indices)
        rebuild_values_func, rebuild_values_args = reduce_tensor(values)

        return (
            rebuild_sparse_compressed_tensor,
            (
                rebuild_compressed_indices_func,
                rebuild_compressed_indices_args,
                rebuild_plain_indices_func,
                rebuild_plain_indices_args,
                rebuild_values_func,
                rebuild_values_args,
                shape,
                layout,
            ),
        )
    else:
        raise RuntimeError(f"Unknown sparse tensor layout: {sparse.layout}")


def fd_id(fd):
    # Returns a tuple which uniquely identifies a file descriptor. In Mac OS,
    # this doesn't work with shared memory handles, which is why we don't
    # support the "file_descriptor" sharing method on that platform.
    stat = os.fstat(fd)
    return (stat.st_ino, stat.st_dev)


def storage_from_cache(cls, key):
    storage_ref = _shared_cache.get(key)
    if storage_ref is None:
        return None
    storage = storage_ref.cdata
    if storage is None:
        return None
    return cls._new_with_storage(storage)


def rebuild_storage_fd(cls, df, size):
    fd = df.detach()
    try:
        storage = cls._new_with_file(fd, size)
        return storage
    except Exception:
        os.close(fd)
        raise


def rebuild_storage_filename(cls, manager, handle, size, dtype=None):
    if dtype is None:
        dtype = cls._scalar_type
    storage = cls._new_with_file(handle, size, dtype)
    return storage


def rebuild_storage_empty(cls):
    return cls._new_with_storage(torch.UntypedStorage._new_empty())


def rebuild_typed_storage(storage, dtype):
    return storage.to(dtype)


def reduce_typed_storage(storage):
    return rebuild_typed_storage, (storage, storage.dtype)


def rebuild_typed_storage_child(storage, storage_type):
    return storage_type._new_with_storage(storage)


def reduce_typed_storage_child(storage):
    return rebuild_typed_storage_child, (storage, type(storage))


def reduce_storage(storage):
    from multiprocess.reduction import ForkingPickler

    if storage.is_cuda:
        # This is a CUDA storage
        storage_handle = storage._share_cuda_()
        storage_size_bytes = storage.size() * storage.element_size()
        return (
            rebuild_cuda_tensor,
            (
                type(storage),
                storage.size(),
                storage.stride(),
                storage.storage_offset(),
                type(storage),
                storage.dtype,
                storage.device,
                storage_handle,
                storage_size_bytes,
                0,
                False,
                None,
                0,
                None,
                False,
            ),
        )
    else:
        # This is a CPU storage
        if storage.is_shared:
            # This is a shared memory storage
            storage_handle = storage._share_filename_()
            storage_size_bytes = storage.size() * storage.element_size()
            return (
                rebuild_cuda_tensor,
                (
                    type(storage),
                    storage.size(),
                    storage.stride(),
                    storage.storage_offset(),
                    type(storage),
                    storage.dtype,
                    storage.device,
                    storage_handle,
                    storage_size_bytes,
                    0,
                    False,
                    None,
                    0,
                    None,
                    False,
                ),
            )
        else:
            # This is a regular CPU storage
            storage_handle = storage._share_filename_()
            storage_size_bytes = storage.size() * storage.element_size()
            return (
                rebuild_cuda_tensor,
                (
                    type(storage),
                    storage.size(),
                    storage.stride(),
                    storage.storage_offset(),
                    type(storage),
                    storage.dtype,
                    storage.device,
                    storage_handle,
                    storage_size_bytes,
                    0,
                    False,
                    None,
                    0,
                    None,
                    False,
                ),
            )


def init_reductions():
    """Register reduction functions for torch tensors."""
    from multiprocess.reduction import ForkingPickler

    ForkingPickler.register(torch.HalfTensor, reduce_tensor)
    ForkingPickler.register(torch.BFloat16Tensor, reduce_tensor)
    ForkingPickler.register(torch.FloatTensor, reduce_tensor)
    ForkingPickler.register(torch.DoubleTensor, reduce_tensor)
    ForkingPickler.register(torch.ShortTensor, reduce_tensor)
    ForkingPickler.register(torch.IntTensor, reduce_tensor)
    ForkingPickler.register(torch.LongTensor, reduce_tensor)
    ForkingPickler.register(torch.CharTensor, reduce_tensor)
    ForkingPickler.register(torch.ByteTensor, reduce_tensor)
    ForkingPickler.register(torch.BoolTensor, reduce_tensor)
    ForkingPickler.register(torch.cuda.HalfTensor, reduce_tensor)
    ForkingPickler.register(torch.cuda.BFloat16Tensor, reduce_tensor)
    ForkingPickler.register(torch.cuda.FloatTensor, reduce_tensor)
    ForkingPickler.register(torch.cuda.DoubleTensor, reduce_tensor)
    ForkingPickler.register(torch.cuda.ShortTensor, reduce_tensor)
    ForkingPickler.register(torch.cuda.IntTensor, reduce_tensor)
    ForkingPickler.register(torch.cuda.LongTensor, reduce_tensor)
    ForkingPickler.register(torch.cuda.CharTensor, reduce_tensor)
    ForkingPickler.register(torch.cuda.ByteTensor, reduce_tensor)
    ForkingPickler.register(torch.cuda.BoolTensor, reduce_tensor)
    ForkingPickler.register(torch.cuda.Event, reduce_event)
    ForkingPickler.register(torch.UntypedStorage, reduce_storage)
    ForkingPickler.register(torch.TypedStorage, reduce_typed_storage)
    ForkingPickler.register(torch._C._NestedTensor, reduce_nested_tensor)
    ForkingPickler.register(torch._C._SparseCooTensor, reduce_sparse_tensor)
    ForkingPickler.register(torch._C._SparseCsrTensor, reduce_sparse_tensor)
    ForkingPickler.register(torch._C._SparseCscTensor, reduce_sparse_tensor)
    ForkingPickler.register(torch._C._SparseBsrTensor, reduce_sparse_tensor)
    ForkingPickler.register(torch._C._SparseBscTensor, reduce_sparse_tensor) 