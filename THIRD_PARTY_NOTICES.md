# Third-party components

The MIT license covers this repository's implementation. It does not relicense
external models, datasets, or dependencies.

- TempAct: https://github.com/jingw193/TempAct — optional LongLive/Wan integration.
- LongLive: https://github.com/NVlabs/LongLive — causal video generation.
- Wan: https://github.com/Wan-Video/Wan2.1 — backbone, VAE, and text encoder.
- verl: https://github.com/verl-project/verl — optional planner training integration.
- VBench: https://github.com/Vchitect/VBench — optional official metric implementations and prompt metadata (Apache-2.0). Evaluation uses the separately installed, provenance-recorded implementation and records local compatibility patches.
- RAVEN: https://arxiv.org/abs/2605.15190 — consistency-policy and reward reference.
- Turns Matter: https://arxiv.org/abs/2611.06728 — decision-level credit reference.
- Flex-Forcing: https://arxiv.org/abs/2607.03509 — variable-block training reference.

Upstream code, pretrained weights, private prompts, and historical run artifacts
are not bundled. The new project follows the design of the earlier internal
TempAct-duration experiments, while replacing their experiment-specific training
contracts and machine-specific configuration.
