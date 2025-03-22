<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/source/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/source/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://vllm.ai"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

---

## vLLM for Windows

This repository is a fork of vLLM and will be updated when new release versions of vLLM are published, until vLLM decides to support Windows oficially (see https://github.com/vllm-project/vllm/issues/14981 / https://github.com/vllm-project/vllm/pull/14891).

**Don't open a new Issue to request a specific commit build. Wait for a new stable release.**

**Don't open Issues for general vLLM questions or non Windows related problems. Only Windows specific issues.** Any Issue opened that is not Windows specific will be closed automatically.

**Don't request a wheel for your specific environment.** Currently, the only wheels I will publish are for Python 3.12 + CUDA 12.4 + torch 2.6.0. If you have another versions, build your own wheel from source following the instructions below.

### Windows instructions:

#### Installing an existing release wheel:

1. Ensure that you have the correct Python and CUDA version of the wheel. The Python and CUDA version of the wheel is specified in the release version
2. Download the wheel from the release version of your preference
3. Install it with ```pip install DOWNLOADED_WHEEL_PATH```

#### Building from source:

A Visual Studio 2019 or newer is required to launch the compiler x64 environment. The installation path is referred in the instructions as VISUAL_STUDIO_INSTALL_PATH.

CUDA path will be found automatically if you have the bin folder in your PATH, or have the CUDA installation path settled on well-known environment vars like CUDA_ROOT, CUDA_HOME or CUDA_PATH.

If none of these are present, make sure to set the environment variable before starting the build:
set CUDA_ROOT=CUDA_INSTALLATION_PATH

1. Open a Command Line (cmd.exe)
2. Clone the vLLM repository: ```cd C:\ & git clone https://github.com/SystemPanic/vllm-windows.git```
3. Execute (in cmd) ```VISUAL_STUDIO_INSTALL_PATH\VC\Auxiliary\Build\vcvarsall.bat x64```
4. Change the working directory to the cloned repository path, for example: ```cd C:\vllm-windows```
5. Set the following environment variables:

```
set DISTUTILS_USE_SDK=1
set VLLM_TARGET_DEVICE=cuda
set MAX_JOBS=10 (or your desired number to speed up compilation)

#Optional variables:

#To include cuDSS (only if you have cuDSS installed)
set USE_CUDSS=1
set CUDSS_LIBRARY_PATH=PATH_TO_CUDSS_INSTALL_DIR\lib\12
set CUDSS_INCLUDE_PATH=PATH_TO_CUDSS_INSTALL_DIR\include

#To include cuSPARSELt (only if you have cuSPARSELt installed)
set CUSPARSELT_INCLUDE_PATH=PATH_TO_CUSPARSELT_INSTALL_DIR\include 
set USE_CUSPARSELT=1

#To include cuDNN:
set USE_CUDNN=1

#Flash Attention v3 build has been disabled inside WSL2 and Windows due to compiler being killed on WSL2, and extremely long compiling times on Windows. Hopper is not available on Windows, so FA3 has no sense anyway. 
#Build can be forcefully enabled using the following environment var:
set VLLM_FORCE_FA3_WINDOWS_BUILD=1

```
6. Build & install:
```
#With vLLM torch version
pip install . --no-build-isolation

#With currently installed torch version
python use_existing_torch.py
pip install -r requirements/build.txt
pip install . --no-build-isolation
```

---

[2025/03] We are collaborating with Ollama to host an [Inference Night](https://lu.ma/vllm-ollama) at Y Combinator in San Francisco on Thursday, March 27, at 6 PM. Discuss all things inference local or data center!

[2025/04] We're hosting our first-ever *vLLM Asia Developer Day* in Singapore on *April 3rd*! This is a full-day event (9 AM - 9 PM SGT) in partnership with SGInnovate, AMD, and Embedded LLM. Meet the vLLM team and learn about LLM inference for RL, MI300X, and more! [Register Now](https://www.sginnovate.com/event/limited-availability-morning-evening-slots-remaining-inaugural-vllm-asia-developer-day)

---

*Latest News* 🔥

- [2025/03] We hosted [the first vLLM China Meetup](https://mp.weixin.qq.com/s/n77GibL2corAtQHtVEAzfg)! Please find the meetup slides from vLLM team [here](https://docs.google.com/presentation/d/1REHvfQMKGnvz6p3Fd23HhSO4c8j5WPGZV0bKYLwnHyQ/edit?usp=sharing).
- [2025/03] We hosted [the East Coast vLLM Meetup](https://lu.ma/7mu4k4xx)! Please find the meetup slides [here](https://docs.google.com/presentation/d/1NHiv8EUFF1NLd3fEYODm56nDmL26lEeXCaDgyDlTsRs/edit#slide=id.g31441846c39_0_0).
- [2025/02] We hosted [the ninth vLLM meetup](https://lu.ma/h7g3kuj9) with Meta! Please find the meetup slides from vLLM team [here](https://docs.google.com/presentation/d/1jzC_PZVXrVNSFVCW-V4cFXb6pn7zZ2CyP_Flwo05aqg/edit?usp=sharing) and AMD [here](https://drive.google.com/file/d/1Zk5qEJIkTmlQ2eQcXQZlljAx3m9s7nwn/view?usp=sharing). The slides from Meta will not be posted.
- [2025/01] We are excited to announce the alpha release of vLLM V1: A major architectural upgrade with 1.7x speedup! Clean code, optimized execution loop, zero-overhead prefix caching, enhanced multimodal support, and more. Please check out our blog post [here](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html).
- [2025/01] We hosted [the eighth vLLM meetup](https://lu.ma/zep56hui) with Google Cloud! Please find the meetup slides from vLLM team [here](https://docs.google.com/presentation/d/1epVkt4Zu8Jz_S5OhEHPc798emsYh2BwYfRuDDVEF7u4/edit?usp=sharing), and Google Cloud team [here](https://drive.google.com/file/d/1h24pHewANyRL11xy5dXUbvRC9F9Kkjix/view?usp=sharing).
- [2024/12] vLLM joins [PyTorch ecosystem](https://pytorch.org/blog/vllm-joins-pytorch)! Easy, Fast, and Cheap LLM Serving for Everyone!

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has evolved into a community-driven project with contributions from both academia and industry.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests
- Fast model execution with CUDA/HIP graph
- Quantizations: [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), INT4, INT8, and FP8.
- Optimized CUDA kernels, including integration with FlashAttention and FlashInfer.
- Speculative decoding
- Chunked prefill

**Performance benchmark**: We include a performance benchmark at the end of [our blog post](https://blog.vllm.ai/2024/09/05/perf-update.html). It compares the performance of vLLM against other LLM serving engines ([TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [SGLang](https://github.com/sgl-project/sglang) and [LMDeploy](https://github.com/InternLM/lmdeploy)). The implementation is under [nightly-benchmarks folder](.buildkite/nightly-benchmarks/) and you can [reproduce](https://github.com/vllm-project/vllm/issues/8176) this benchmark using our one-click runnable script.

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor parallelism and pipeline parallelism support for distributed inference
- Streaming outputs
- OpenAI-compatible API server
- Support NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, TPU, and AWS Neuron.
- Prefix caching support
- Multi-lora support

vLLM seamlessly supports most popular open-source models on HuggingFace, including:
- Transformer-like LLMs (e.g., Llama)
- Mixture-of-Expert LLMs (e.g., Mixtral, Deepseek-V2 and V3)
- Embedding Models (e.g. E5-Mistral)
- Multi-modal LLMs (e.g., LLaVA)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with `pip` or [from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source):

```bash
pip install vllm
```

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.
- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [CONTRIBUTING.md](./CONTRIBUTING.md) for how to get involved.

## Sponsors

vLLM is a community project. Our compute resources for development and testing are supported by the following organizations. Thank you for your support!

<!-- Note: Please sort them in alphabetical order. -->
<!-- Note: Please keep these consistent with docs/source/community/sponsors.md -->
Cash Donations:
- a16z
- Dropbox
- Sequoia Capital
- Skywork AI
- ZhenFund

Compute Resources:
- AMD
- Anyscale
- AWS
- Crusoe Cloud
- Databricks
- DeepInfra
- Google Cloud
- Lambda Lab
- Nebius
- Novita AI
- NVIDIA
- Replicate
- Roblox
- RunPod
- Trainy
- UC Berkeley
- UC San Diego

Slack Sponsor: Anyscale

We also have an official fundraising venue through [OpenCollective](https://opencollective.com/vllm). We plan to use the fund to support the development, maintenance, and adoption of vLLM.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues) or [Discussions](https://github.com/vllm-project/vllm/discussions)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu)

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit).
