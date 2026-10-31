# Motor do rec com o Flint em especialistas de 3 bits no M4 Pro 24 GB

**Timestamp:** 2026-10-30T19:25
**Branch:** `mac-apple-silicon`
**Plano seguido:** passos 1–5 de [2026-10-30-estado-sessao-e-proximos-passos.md](2026-10-30-estado-sessao-e-proximos-passos.md)

## Contexto

- **Pedido do usuário:** "tente utilizar a mesma logica de otimização do rec (https://github.com/recursive-oss/rec) para o mlx
  ou llama.cpp para melhorar a velocidade de inferencia". Plano aprovado com "Sim, siga o plano todo"; retomado com "continue".
- **Origem das ideias:**
  - **Passado pelo usuário:** o motor `rec` ("eles usam um motor de inferencia otimizado que se chama rec"), a busca por forks
    do rec para Mac e, nesta sessão, as perguntas sobre outra quantização e sobre o `UD-Q2_K_XL`.
  - **Proposto pelo Claude:** usar o fork `thinkspread/rec@local/flint-e3-24gb` com um checkpoint de especialistas em 3 bits
    gerado aqui (registrado no arquivo de estado antes da compactação); nesta sessão, a busca de escala por erro quadrático e
    o patch do bloco de KV.

## Diagnóstico

- O motor do rec (megakernel Metal residente, kernels de especialistas e de KV) é o que dá 125+ tok/s no M4 Pro de 48 GB, mas o
  checkpoint oficial tem 22,6 GB e o CLI recusa Macs com menos de 48 GiB.
- O fork só tira a trava e aceita `routed_expert_bits` 3. O carregador (`quantized.py`), o megakernel
  (`assert p.bits in (3, 4, 8)`) e os kernels de bits mistos do upstream já tratavam 3 bits. Faltava o checkpoint.
- **Primeira carga:** o motor reserva o KV em blocos fixos de 65.536 tokens. Isso dá 1,78 GiB já no primeiro pedido, e o
  tamanho está gravado no kernel (`(i/128)>>16`, `65536u*128u`). Os pesos (17,0 GiB) mais esse bloco passaram do que o Mac tem
  livre (~19 GB utilizáveis), e o primeiro pedido empurrou 3.057 MiB para o swap. O vigia do launcher parou o servidor.
- O `memory_pressure` dizia "85% livre", mas o `vm_stat` mostrava 17,1 GB livres mais 2 GB inativos. O número dele exagera a
  folga real.

## Alterações

- **`mac/rec_requant_e3.py`** (novo):
  - lê cada shard do `RecursiveSI/flint-int4` tensor por tensor e muda só os `.switch_mlp.` (INT4 g64 para INT3 g64);
  - a escala de cada grupo sai de uma busca de 5×5 cortes nas pontas da faixa (0–40%), ficando com o de menor erro quadrático;
  - grava os pesos em U8 como o original, escreve `.partial` e renomeia, e com `--delete-src` apaga cada shard de origem;
  - ajusta o `config.json` (`routed_expert_bits` 3) e gera o manifesto `flint-int4-e3.json` com SHA-256 de todos os arquivos.
- **Escolha da busca** (camada 22, erro relativo ao INT4): `mx.quantize` simples 3,46%; corte simétrico 2,99%; 5×5 assimétrico
  2,88%; 9×9 2,85% a três vezes o custo. Ficou o 5×5.
- **`mac/rec-e3-kv16k.patch`** (novo, aplicado em `~/src/rec-e3`):
  - `cache.py`: `BLOCK_TOKENS` de 65536 para 16384;
  - `kernels/megakernel.py`: `(i/128)>>14` e `16384u*128u` no endereço do KV;
  - `engine.py`: `mx.set_cache_limit` de 2 GiB para 0.
- **`mac/serve_rec.sh`** (novo): launcher com o preflight e o vigia de `mac/common.sh`. Lê o tamanho do bloco do motor
  instalado para estimar o KV e passa a chave da API por arquivo temporário (modo 600) quando existe uma.
- **`mac/common.sh`:** a lista de servidores passou a incluir `bin/rec-mlx` e `bin/tempo9`. Ela casa pelo caminho `bin/` porque
  a primeira versão (`rec-mlx`) casou com a linha de comando do meu próprio monitor e bloqueou a carga.
- **`mac/README.md`:** seção "rec com especialistas em 3 bits", e os trechos que diziam que o rec não roda aqui foram ajustados.
- **Fora do repositório:**
  - `~/models/flint-int4-e3` (17,0 GiB); de `~/models/flint-int4` ficaram só os arquivos pequenos;
  - `~/src/rec-e3` (fork, commit 325d244 mais o patch);
  - `~/.venvs/rec-e3` (Python 3.12, mlx 0.32.0, rec-mlx).

## Resultados

Mesmo protocolo das tabelas do `mac/README.md`, com os padrões de sorteio do rec (temperatura 1,0, top-p 0,95, top-k 20):

| Flint | chat (tok/s) | cópia (tok/s) | cópia: tokens / cobertura |
|---|---|---|---|
| **rec, especialistas INT3, bloco de KV de 16k** | **101,52** | 88,71 | 4.911 / 0,92 |
| llama.cpp UD-IQ3_XXS, sem drafter | 62,67 | 56,61 | 7.850 / 0,99 |
| llama.cpp UD-IQ3_XXS, só lookup | 62,26 | 121,78 | 6.972 / 0,98 |
| llama.cpp UD-IQ3_XXS, DFlash2 3 drafts + lookup | 65,84 | 118,19 | 6.798 / 0,92 |

- **Chat:** 1,5× o melhor llama.cpp. Os 8 pedidos ficaram entre 99,9 e 102,6 tok/s; o motor mediu 98–101 tok/s.
- **Cópia:** 88,7 tok/s, abaixo do lookup do llama.cpp (121,8), porque o rec não especula.
- **Tentativas de carga:**

  | Tentativa | Pico do processo | Swap |
  |---|---|---|
  | bloco de 65k (upstream), cache 2 GiB | 19,46 GB na carga, mais 1,8 GB de KV no primeiro pedido | +3.057 MiB, vigia parou |
  | bloco de 16k, cache 512 MB, RustDesk e Monitor de Atividade abertos | 19,06 GB | +548 MB em 3 pedidos, vigia parou |
  | **bloco de 16k, cache 0, esses apps fechados** | 19,1 GB no chat, 19,85 GB na cópia | **0** em 10 pedidos |

- Carga em ~5 s, mais o empacotamento dos pesos, com pico de 19,46 GB no MLX nessa etapa.
- A resposta de teste ("Qual é a capital da Austrália?") saiu correta: "A capital da Austrália é Canberra."
- O LM Studio apareceu aberto no meio da sessão, mais de uma hora depois de o usuário recusar ativá-lo. Não fui eu que abri;
  origem a confirmar. O usuário fechou o LM Studio, o RustDesk e o Monitor de Atividade antes da última carga.
- Arquivos: `mac/results/m4pro-rec-flint-e3-kv16k.json` e `mac/results/m4pro-rec-flint-e3-kv16k.engine-metrics.jsonl`
  (métricas por pedido do motor, sem os ids dos tokens).

## Issues remanescentes

- **Qualidade não medida em benchmark.** Falta um teste (GSM8K ou as tarefas do usuário) comparando INT3 com o llama.cpp IQ3_XXS.
- **Memória no limite.** Só roda com LM Studio, Chrome, RustDesk e Monitor de Atividade fechados. Contexto maior que 16k soma
  0,45 GiB por bloco.
- **Patch do kernel sem teste de contexto longo.** Os pedidos foram até ~7 mil tokens, dentro do primeiro bloco. Contextos que
  passam de um bloco (mais de 16.384 tokens) usam o novo endereço `>>14` e ainda não foram testados.
- **`bench.py` com o rec:** o `finish_reason` em streaming vem vazio (`None`).
- **Outras quantizações não servem no rec:** o kernel aceita só 3, 4 ou 8 bits e grupos de 64. O `UD-Q2_K_XL` (13,6 GB) é GGUF
  e só roda no llama.cpp; lá a estimativa é de 66–70 tok/s, com qualidade menor. Não testado.
- Os commits `30e7638`, `125c075` e as mudanças desta sessão ainda não estão na `main` do fork.

## Pi (pi.dev) com o rec

- **Pedido do usuário:** "Seria possivel utilizar esse modelo com o pi.dev?" e depois "monte o .sh para testar".
- **`mac/pi_rec.sh`** (novo): faz o que o `rec launch pi` faz, sem o CLI do rec.
  - Instala o `@mariozechner/pi-coding-agent@0.73.1` com `npm ci` a partir do `package-lock.json` do fork, em
    `~/.local/share/hyperqwen-pi` (fora do Drive).
  - Copia a extensão do rec (`flint-pi.ts`, `pi-transport.mjs`, `pi-logo.mjs` e o logo) e abre o pi com
    `--provider rec --model flint`, apontado para o `serve_rec.sh` na porta 18020, com contexto de 16k.
- **Não testado ainda:** o classificador do modo automático bloqueou o `npm ci` rodado pelo Claude. A instalação fica com o
  usuário (`INSTALL_ONLY=1 bash mac/pi_rec.sh`).

## Arquivos tocados

- `mac/pi_rec.sh` (novo)
- `mac/rec_requant_e3.py` (novo)
- `mac/rec-e3-kv16k.patch` (novo)
- `mac/serve_rec.sh` (novo)
- `mac/common.sh`
- `mac/README.md`
- `mac/results/m4pro-rec-flint-e3-kv16k.json` (novo)
- `mac/results/m4pro-rec-flint-e3-kv16k.engine-metrics.jsonl` (novo)
- `Atualizações/2026-10-30-rec-flint-int3-m4pro.md` (este arquivo)
- `Atualizações/README.md`
