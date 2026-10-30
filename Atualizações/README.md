# Atualizações

Registro cronológico das sessões que alteram código, configuração ou dados neste clone do HyperQwen.

| Data | Registro | Resumo |
|---|---|---|
| 2026-10-30 | [2026-10-30-mac-m4pro-llamacpp-dflash2.md](2026-10-30-mac-m4pro-llamacpp-dflash2.md) | Qwen3.8-27B no M4 Pro 24 GB com llama.cpp 0.8.0: DFlash2 + lookup fazem 18 tok/s no chat e 38 tok/s na cópia (sem especulação: 10). MLX: o melhor é o oMLX com DFlash2 (17,6 no chat, 27,5 na cópia); o mlx_vlm com MTP faz 17,3. Recursive Flint (MoE, outro modelo) no llama.cpp: 62 tok/s no chat, 122 na cópia. |
