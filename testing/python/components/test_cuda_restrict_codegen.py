import tilelang
import tilelang.language as T
import tilelang.testing


def _get_sig_line(code: str) -> str:
    # Find the kernel signature line in generated CUDA code
    for line in code.splitlines():
        line = line.strip()
        if line.startswith('extern "C" __global__ void'):
            return line
    raise AssertionError("Kernel signature not found in generated code")


@tilelang.testing.requires_cuda
def test_cuda_restrict_default_has_restrict():
    N = 128

    @T.prim_func
    def kernel(x: T.Tensor((N,), T.float32), y: T.Tensor((N,), T.float32)):
        with T.Kernel(N, threads=32) as pid:
            y[pid] = x[pid] + 1.0

    artifact = tilelang.lower(kernel, target="cuda")
    sig = _get_sig_line(artifact.kernel_source)
    # By default, kNoAlias is set and both pointers are restrict-qualified
    assert "__restrict__" in sig


@tilelang.testing.requires_cuda
def test_cuda_restrict_annotation_removes_restrict():
    N = 128

    @T.prim_func
    def kernel_body_annot(x: T.Tensor((N,), T.float32), y: T.Tensor((N,), T.float32)):
        # Explicitly mark buffers that may alias as non-restrict
        with T.Kernel(N, threads=32) as pid:
            T.annotate_restrict_buffers(x, y)
            y[pid] = x[pid] + 1.0

    art1 = tilelang.lower(kernel_body_annot, target="cuda")
    sig1 = _get_sig_line(art1.kernel_source)
    # No parameter should be emitted with __restrict__
    assert "__restrict__" not in sig1


@tilelang.testing.requires_cuda
def test_cuda_restrict_annotation_keeps_handle_suffix_buffers():
    N = 128

    @T.prim_func
    def kernel_body_annot(A: T.Tensor((N,), T.float32), A_handle: T.Tensor((N,), T.float32)):
        with T.Kernel(N, threads=32) as pid:
            T.annotate_restrict_buffers(A, A_handle)
            A_handle[pid] = A[pid] + 1.0

    artifact = tilelang.lower(kernel_body_annot, target="cuda")
    sig = _get_sig_line(artifact.kernel_source)
    assert "__restrict__" not in sig


def test_nonrestrict_hoist_preserves_same_hint_var_identity():
    tirx = tilelang.tvm.tirx
    a = tirx.Var("A", "handle")
    b = tirx.Var("A", "handle")
    block = tirx.SBlock(
        [],
        [],
        [],
        "root",
        tirx.Evaluate(0),
        annotations={"tl.non_restrict_params": [a, b, a]},
    )
    func = tirx.PrimFunc([a, b], tirx.SBlockRealize([], True, block))
    func = func.with_attr("tl.non_restrict_params", [b, a])
    mod = tilelang.transform.HoistNonRestrictParams()(tilelang.tvm.IRModule({"main": func}))
    variables = mod["main"].attrs["tl.non_restrict_params"]
    assert len(variables) == 2
    assert variables[0].same_as(a)
    assert variables[1].same_as(b)


def test_nonrestrict_function_attribute_preserves_same_hint_var_identity():
    tirx = tilelang.tvm.tirx
    a = tirx.Var("A", "handle")
    b = tirx.Var("A", "handle")
    func = tirx.PrimFunc([a, b], tirx.Evaluate(0)).with_attr("tl.non_restrict_params", [a, b, a])
    mod = tilelang.transform.HoistNonRestrictParams()(tilelang.tvm.IRModule({"main": func}))
    variables = mod["main"].attrs["tl.non_restrict_params"]
    assert len(variables) == 2
    assert variables[0].same_as(a)
    assert variables[1].same_as(b)


if __name__ == "__main__":
    tilelang.testing.main()
