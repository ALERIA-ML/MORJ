# NVIDIA's TensorRT-LLM release image for the validated version. It ships torch,
# FlashInfer, the NVFP4 GEMM kernels for Blackwell, MPI, nvcc and the HTTP stack;
# the service adds only its own code.
FROM nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc29
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /app
COPY morj /app/morj
# Fail the build unless TRT-LLM matches the sources the patches were validated against.
RUN python -c "from morj.patches import verify_runtime; verify_runtime()"
CMD ["python", "-m", "morj"]
