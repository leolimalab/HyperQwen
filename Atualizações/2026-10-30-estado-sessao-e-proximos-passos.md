# Estado da sessão e próximos passos (handoff antes da compactação)

**Timestamp:** 2026-10-30T18:08
**Pedido do usuário:** "armazene o estado dessa sessão e faça uma compactação de contexto" antes de seguir o plano do rec com
especialistas em 3 bits (aprovado pelo usuário: "Sim, siga o plano todo").
**Registro completo da sessão:** [2026-10-30-mac-m4pro-llamacpp-dflash2.md](2026-10-30-mac-m4pro-llamacpp-dflash2.md)

## Onde as coisas estão

- **Repositório:** branch `mac-apple-silicon`, que acompanha `leolimalab/mac-apple-silicon`.
  - Último commit enviado: `30e7638`. A `main` do fork (`a3dc382`, merge do PR #1) **não** tem esse commit; falta um PR novo ou um merge.
  - Árvore de trabalho limpa, exceto por este arquivo.
- **Remotes:** `origin` = syv-ai/HyperQwen (original); `leolimalab` = fork público do usuário.
- **Mac:** MacBook Pro M4 Pro, 24 GB, macOS 26.6.2.
  - `iogpu.wired_limit_mb=20480`, que volta ao padrão no próximo reboot.
  - Disco: 45 GB livres.
- **Software:**
  - llama.cpp 0.8.0 (build 11960, Homebrew);
  - pyenv global Python 3.10.14 com mlx 0.32.4, mlx-vlm 0.7.11 e mlx-lm 0.31.3 (a 0.32 exige Python 3.11+);
  - oMLX 0.6.3rc1 instalado (app);
  - tempo9 **desinstalado**; o tap `thinkspread/tap` continua.
- **Serviços:**
  - LM Studio server ligado (porta 1234, sem modelo; o Qwen 3.5 9B carrega sob demanda para o ai-memory);
  - ai-memory ligado (405) e rodando a "melhoria automática" nos outros projetos.

## Modelos em disco (fora do Drive)

| Caminho | Uso |
|---|---|
| `~/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS.gguf` (14,25 GB) + `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` | Qwen, `bash mac/serve.sh` |
| `~/.lmstudio/models/incoai/Qwen3.8-27B-DFlash2-GGUF/…Q4_K_M.gguf` (rev. 144eb06, com o mrope fix) | drafter do Qwen |
| `~/.lmstudio/models/unsloth/flint-GGUF/flint-UD-IQ3_XXS.gguf` (14,5 GB) | Flint, `MODEL_PRESET=flint bash mac/serve.sh` |
| `~/.lmstudio/models/incoai/Flint-DFlash2-GGUF/Flint-DFlash2-Q8_0.gguf` (0,43 GB) | drafter do Flint (padrão do preset: 3 drafts) |
| `~/models/mlx-drafters/` (DFlash2 e MTP em MLX) | `serve_mlx.sh` / `serve_omlx.sh`. O alvo MLX só texto foi apagado; recriar com `bash mac/download.sh mlx-text` |

## Melhores números medidos (M4 Pro 24 GB, um stream, tok/s chat / cópia)

| Configuração | chat | cópia |
|---|---|---|
| Qwen3.8-27B, llama.cpp, DFlash2 + lookup (padrão) | 18,05 | 38,28 |
| Qwen3.8-27B, oMLX, DFlash2 bloco 8 adaptativo | 17,62 | 27,53 |
| Flint UD-IQ3_XXS, llama.cpp, só lookup | 62,26 | 121,78 |
| Flint UD-IQ3_XXS, llama.cpp, DFlash2 3 drafts + lookup (padrão do preset) | 65,84 | 118,19 |

O que não funciona neste Mac:
- `rec` oficial: exige 48 GiB;
- `tempo9`: GPU só em M5; na CPU dá ~17 tok/s, e com `AS_METAL_TENSOR_FORCE=1` gera lixo;
- MLX com drafter: só funciona com `LOAD_GRACE_S`;
- GGUF Q6K de 18 GB no llama.cpp: deixou 8% de memória livre.

## Regras do usuário que valem aqui

- Antes de **cada** carga de modelo pesado: AskUserQuestion dizendo quantos GB usa e o que sai da memória, com LM Studio e Chrome
  fechados. O Zed fica aberto, porque é onde a sessão roda.
- Parar na hora se o swap subir ou a velocidade cair. Os scripts têm um vigia (`SWAP_LIMIT_MB`), e o script de testes cancela
  quando sobra menos de 12% livre depois de carregar.
- Nunca chamar `lms ps` com o LM Studio fechado, porque acorda o serviço. Nunca descarregar modelos do LM Studio.
- Commit e push só quando o usuário pedir. Registrar tudo em `Atualizações/`.

## Próximo passo aprovado: motor do rec (MLX) com o Flint em especialistas de 3 bits

**Por quê:** o motor do rec (megakernel Metal, workers residentes, kernels de especialistas e de KV) é o que dá 125+ tok/s no
M4 Pro de 48 GB. O upstream **já suporta** `routed_expert_bits` 3:
- `engine/src/rec_mlx/quantized.py` aceita 3 ou 4;
- `kernels/megakernel.py` faz `assert p.bits in (3, 4, 8)`;
- `kernels/mixed_bits.py` tem um caminho para `bits == 3`.

O fork `thinkspread/rec@local/flint-e3-24gb` (commits 4eacf0ad e 325d2446) só tira a trava de 48 GiB do `engine.py`, relaxa o
`checkpoint.py` (especialistas 3/4) e chama `mx.set_wired_limit(max_recommended_working_set_size)`. O checkpoint `flint-int4-e3`
não está publicado: é preciso gerá-lo.

**Passos:**
1. Baixar `RecursiveSI/flint-int4` (MLX, 22,57 GB, 5 shards `model-0000{0..4}.safetensors`, mais `config.json`, `tokenizer*`,
   `special_tokens_map.json`, `chat_template.jinja`, `generation_config.json` e `flint-int4.json`) para uma pasta fora do Drive,
   por exemplo `~/models/flint-int4`.
2. Converter **shard por shard**, apagando cada original depois de gravar o novo (pico de disco de ~27 GB):
   - nos tensores com `.switch_mlp.` no nome (`weight`, `scales`, `biases`): `mx.dequantize(bits=4, group_size=64)` seguido de
     `mx.quantize(bits=3, group_size=64)`;
   - o resto fica igual (densas em INT4 g64, `model.embed_tokens` em INT8);
   - no `config.json`, `mlx_export.routed_expert_bits` passa de 4 para 3, e `quantization` continua `{"group_size": 64, "bits": 4}`;
   - o original usa `scale_rule: mse`, e o `mx.quantize` é min-max; se a qualidade cair demais, buscar escalas por MSE.
   - Resultado esperado: ~17,9 GB, com 1,39 GB lidos por token.
3. Manifesto próprio, no formato de `flint-int4.json`: `weight_format: "mlx-affine-int4"`, `kv_format: "bf16"` e
   `checkpoint_files` listando **todos** os arquivos (o `checkpoint.py` exige `config.json`, `tokenizer.json`,
   `tokenizer_config.json` e `special_tokens_map.json`, e os shards precisam bater com o glob).
4. Motor: clonar `thinkspread/rec` (branch `local/flint-e3-24gb`) fora do Drive. Venv com **Python ≥ 3.12** via `uv` (fora do
   Drive), com `mlx==0.32.0`, `mlx-lm==0.31.3`, `transformers==5.16.1`, `xgrammar==0.2.1` e o build nativo (`nanobind==2.13.0`,
   `engine/native/build.py`). Instalar `engine/` e usar o comando `rec-mlx flint --model-dir <dir> --manifest <json> …`
   (ver `engine/src/rec_mlx/cli.py` para porta e host).
5. Launcher `mac/serve_rec.sh` (preflight e vigia de `mac/common.sh`) e benchmark com `mac/bench.py --model flint --quote-max-tokens 8000`.
   O rec quantiza o KV em 8/4 bits (`MixedQuantizedKVCache`). Pedir aval antes de carregar (~19 GB) e comparar com os 62–66 tok/s
   do llama.cpp.

**Riscos:**
- A memória fica perto do limite. Pode exigir baixar a trava de 12% de memória livre depois da carga, sempre com o aval do usuário.
- A qualidade cai com a requantização de 4 para 3 bits; conferir a cobertura da cópia e uma amostra de respostas.
- O motor exige Python 3.12+ e compilação nativa.

## Pendências

- Abrir PR ou fazer merge do `30e7638` para a `main` do fork, se o usuário quiser.
- Qualidade não medida (GSM8K) do Flint IQ3 e do Qwen IQ4_XS no Mac.
- Ajustes finos do Qwen não feitos: `LOOKUP_MAX` 24–32, prompts em português, `-ub` para o prefill.
