# Qwen3.8-27B num Mac Apple Silicon (M4 Pro, 24 GB)

[← README principal](../README.md)

O HyperQwen é vLLM + CUDA de ponta a ponta, e nada disso roda no Metal. Esta pasta
leva as ideias que dão velocidade ao modo single-user para o **llama.cpp no
Metal**, que já tem equivalentes nativos para a maioria delas.

## O que veio do HyperQwen e o que ficou de fora

| Ideia no HyperQwen (3090) | No Mac | Como |
|---|---|---|
| Drafter DFlash2 em bloco, 7 drafts por passada (`SPEC=dflash2`) | **sim** | `--spec-type draft-dflash` com o mesmo checkpoint (z-lab/incoai Qwen3.8-27B-DFlash2, bloco de 8) |
| Drafts tirados do próprio contexto quando a resposta copia o prompt (`LOOKUP=1`, `dflash2-lookup-drafting.patch`) | **sim** | `ngram-mod` junto com o DFlash2. O llama.cpp dá prioridade ao n-gram quando ele dispara, igual ao `LOOKUP_ADAPTIVE` |
| MTP do próprio Qwen (`SPEC=mtp`, 3 drafts) | **sim** | `--spec-type draft-mtp` com o GGUF `MTP/` da unsloth |
| KV em int8/fp8 (`CTX=long`) | **sim** | `-ctk q8_0 -ctv q8_0` |
| Prefix caching em modelo híbrido | **sim** | cache de prompt (`--cache-ram`) com checkpoints do estado recorrente (`--ctx-checkpoints`) |
| lm_head e embeddings requantizados | já vem | o GGUF IQ4_XS já quantiza `token_embd` e `output` |
| Vocabulário reduzido para o drafter (40k ids) | não | o llama.cpp ainda não tem `d2t` para qwen35 (PR #29143 em aberto) |
| Marlin W4A8, FA2 fp8, pth-flash-attn, Triton split-KV | não | são kernels CUDA; o Metal tem os próprios |
| Estado recorrente em 16 bits | não | o llama.cpp guarda o estado GDN em fp32 (~150 MB por sequência) |

## Mais de 60 tok/s: Recursive Flint (outro modelo)

O Qwen3.8-27B é denso: cada token lê os ~14 GB de pesos, e o teto deste Mac fica perto de 19 tok/s sem especulação
e de 18 com ela, no chat. Os "125+ tok/s no M4 Pro" que a Recursive anuncia são de **outro modelo**, o
[Recursive Flint](https://recursive.com/updates/introducing-recursive-flint). Ele é um MoE de 40B no total com 2,5B ativos
por token. O número deles foi medido com o motor próprio (`rec`) num M4 Pro de **48 GB**, em 4 bits.

O `rec` oficial não roda aqui. O manifesto dele exige `min_ram_bytes` de 48 GiB, e o CLI recusa abaixo disso
("this model requires 48 GiB of memory"). O único modelo dele é o `flint-int4` em MLX, de 22,6 GB, que também não cabe
em 24 GB. Testado em 2026-10-30: o instalador oficial (rec 0.1.1) para antes de instalar, com "this model requires 48 GiB of
memory; detected 24 GiB". O **motor** do rec, porém, roda com os especialistas requantizados para 3 bits e faz ~100 tok/s
no chat (seção [rec com especialistas em 3 bits](#rec-com-especialistas-em-3-bits-100-toks-no-chat)). Sem preparar nada,
o caminho é o llama.cpp 0.8.0, que já conhece a arquitetura `flint`, com o `unsloth/flint-GGUF` UD-IQ3_XXS (14,5 GB):

```bash
bash mac/download.sh flint
bash mac/download.sh flint-dflash                # drafter DFlash2 do Flint, 0,43 GB
MODEL_PRESET=flint bash mac/serve.sh             # DFlash2 com 3 drafts + lookup, CTX=long (64k)
MODEL_PRESET=flint SPEC=off bash mac/serve.sh    # sem drafter (0,4 GB a menos)
python3 mac/bench.py flint --model flint --quote-max-tokens 8000
```

Mesmo protocolo da tabela de resultados (2026-10-30, llama.cpp 0.8.0, KV q8_0, 64k de contexto, ~16,6 GB no total,
16–19% de memória livre e sem swap):

| Flint UD-IQ3_XXS | chat (tok/s) | aceitação no chat | cópia (tok/s) | aceitação na cópia | cópia: tokens / cobertura |
|---|---|---|---|---|---|
| sem drafter, sem lookup | 62,67 | — | 56,61 | — | 7.850 / 0,99 |
| só lookup (`SPEC=off`) | 62,26 | 0,34 | **121,78** | 0,95 | 6.972 / 0,98 |
| DFlash2 2 drafts + lookup | 60,03 | 0,48 | 109,77 | 0,86 | 8.000 (limite) / — |
| **DFlash2 3 drafts + lookup (padrão do preset)** | **65,84** | 0,45 | 118,19 | 0,86 | 6.798 / 0,92 |
| DFlash2 3 drafts, sorteio `greedy` + lookup | 60,07 | 0,39 | 104,50 | 0,80 | 7.569 / 0,89 |
| DFlash2 4 drafts + lookup | 63,17 | 0,36 | 106,85 | 0,80 | 8.000 (limite) / — |
| DFlash2 5 drafts + lookup | 62,45 | 0,33 | 111,21 | 0,79 | 8.000 (limite) / — |
| DFlash2 7 drafts + lookup | 54,86 | 0,25 | 105,94 | 0,72 | 8.000 (limite) / — |
| Qwen3.8-27B, DFlash2 + lookup (padrão), para comparar | 18,05 | 74 | 38,28 | 0,96 | 2.335 / 1,0 |

- **3,5× o Qwen no chat e 3,4× na leitura do prompt.** A velocidade não cai em outros idiomas: sem drafter, não há aceitação para cair.
- **O Flint sempre pensa antes de responder.** O template dele não tem chave para desligar o raciocínio. Na cópia, ele
  escreve ~3× o documento em raciocínio antes de copiar. Com 2.400 tokens de limite, ele não chegava à resposta
  (cobertura 0), e foi preciso `--quote-max-tokens 8000`. Por isso, em tempo total, a cópia empata com a do Qwen
  (~57 s contra ~61 s), mesmo com 3× mais tokens por segundo.
- **Lookup:** no chat quase não dispara (aceitação de 0,34 em poucos drafts). Na cópia, dobra a velocidade (121,8 contra
  56,6 tok/s), porque o raciocínio repete o documento.
- **DFlash2 do Flint** (`incoai/Flint-DFlash2-GGUF` Q8_0, 0,43 GB, `bash mac/download.sh flint-dflash`): só rende com bloco curto.
  - Num MoE, verificar k tokens acorda até 6 especialistas por token em cada camada, então o custo da verificação cresce rápido.
  - A aceitação aqui (0,25–0,48) também fica bem abaixo da do cartão do modelo, de 3,5–4,5 tokens por passada em GPU com o
    alvo em BF16. Aqui o alvo está em IQ3.
  - O melhor foi 3 drafts com sorteio `probabilistic`: +5% no chat (65,8 contra 62,7). Isso é perto da variação entre execuções
    com temperatura 1,0, e só uma repetição confirmaria.
  - 7 drafts pioram (54,9). Na cópia, o lookup sozinho continua o melhor.
- **"Limite" na tabela:** o raciocínio muda de tamanho a cada execução (temperatura 1,0). Em quatro configurações ele passou
  de ~27 mil caracteres, e a cópia não terminou dentro dos 8.000 tokens. A velocidade dessas linhas vale como velocidade
  de geração, mas não como cópia completa.
- **Qualidade não foi medida.** O Flint é focado em código agêntico, e o IQ3_XXS perde algo em relação ao NVFP4 oficial.
  Vale um teste de qualidade (GSM8K, ou as suas próprias tarefas) antes de trocar o modelo do dia a dia.

### Forks do rec e Tempo9 (testado em 2026-10-30)

- Dos 43 forks do `recursive-oss/rec`, só um mira Macs de 24 GB: `thinkspread/rec@local/flint-e3-24gb`. Ele tira a trava
  de 48 GiB, aceita especialistas em INT3 e fixa os pesos com `mx.set_wired_limit`. Mas exige um checkpoint MLX `flint-int4-e3`
  que não está publicado em lugar nenhum. O `mac/rec_requant_e3.py` gera esse checkpoint (seção seguinte).
- Os mesmos autores (ThinkSpread) publicam o **Tempo9** e um GGUF de 24 GB do Flint, o `thinkspread/Flint-Q6K-IQ3XXS-GGUF`
  (18,0 GB). Eles anunciam 123–127 tok/s num **M5 Pro** de 24 GB. No **M4 Pro**:
  - o Tempo9 1.2.1 roda **na CPU**: ~17 tok/s. O motor Metal exige a tensor API do Metal 4 (M5 ou mais novo) e
    "no pre-M5 fallback by design";
  - com `AS_METAL_TENSOR_FORCE=1`, ele usa a GPU, mas gera lixo, a ~67 tok/s de texto sem sentido;
  - o GGUF de 18 GB no llama.cpp deixou só 8% de memória livre depois de carregar, e a trava do teste (12%) cancelou a medição.
- O binário do Tempo9 é assinado ("Developer ID Application: Jiejing Zhang") e notarizado pela Apple. Durante os testes,
  só abriu o próprio servidor em `127.0.0.1`. O projeto é muito novo (domínio e organização no GitHub de 17/08/2026, um único
  desenvolvedor), e o motor é fechado.
- Conclusão sobre o Tempo9: no M4 Pro ele não serve. Para o Flint num M4 Pro de 24 GB, ficam o motor do rec com
  especialistas em 3 bits (seção seguinte) e o llama.cpp com o UD-IQ3_XXS (`MODEL_PRESET=flint`).

### rec com especialistas em 3 bits: ~100 tok/s no chat

Testado em 2026-10-30. Usa o motor MLX do rec (megakernel Metal residente) pelo fork `thinkspread/rec@local/flint-e3-24gb`,
com um checkpoint feito aqui a partir do oficial:

- **Checkpoint:** `mac/rec_requant_e3.py` parte do `RecursiveSI/flint-int4` e muda só os especialistas roteados (21,3 dos 22,6 GB): INT4 g64 vira
  INT3 g64. As camadas densas ficam em INT4 e os embeddings em INT8. A escala de cada grupo é escolhida por erro quadrático,
  como o `scale_rule: mse` do original, cortando cada ponta da faixa em 0–40%. O erro em relação ao INT4 fica em 2,89%,
  contra 3,47% do `mx.quantize` simples. O resultado tem 17,0 GiB (5 shards), com manifesto próprio
  (`flint-int4-e3.json`) que passa no `checkpoint.validate` do motor.
- **Bloco de KV:** o motor reserva o KV em blocos fixos de 65.536 tokens, 1,78 GiB já no primeiro pedido, e esse tamanho está
  gravado no kernel Metal. Com ele, o primeiro pedido empurrou 3 GB para o swap. O `mac/rec-e3-kv16k.patch` baixa o bloco
  para 16.384 tokens (0,48 GiB, até 4 blocos, ou seja, 65.536 tokens de contexto) e zera o cache do alocador do MLX.
- **Memória:** pico de 19,1 GB no chat e de 19,85 GB na cópia (7 mil tokens de contexto). Só cabe com LM Studio, Chrome,
  RustDesk e Monitor de Atividade fechados. Com o cache do alocador em 512 MB e o RustDesk e o Monitor de Atividade abertos,
  o swap subiu 548 MB em três pedidos e o vigia parou o servidor. Com o cache zerado e os apps fechados, o swap ficou parado
  nos 10 pedidos.

```bash
git clone --branch local/flint-e3-24gb https://github.com/thinkspread/rec.git ~/src/rec-e3
git -C ~/src/rec-e3 apply "$PWD/mac/rec-e3-kv16k.patch"
uv venv ~/.venvs/rec-e3 --python 3.12
uv pip install --python ~/.venvs/rec-e3/bin/python ~/src/rec-e3/engine
hf download RecursiveSI/flint-int4 --local-dir ~/models/flint-int4            # 22,6 GB
~/.venvs/rec-e3/bin/python -I mac/rec_requant_e3.py ~/models/flint-int4 ~/models/flint-int4-e3 --delete-src   # ~7 min
SWAP_LIMIT_MB=512 LOAD_GRACE_S=60 LOAD_GRACE_MB=1024 bash mac/serve_rec.sh    # "flint" na porta 18020, contexto de 16k
python3 mac/bench.py rec --model flint --stream --quote-max-tokens 8000       # depois que a janela de carga fechar
```

Mesmo protocolo das tabelas acima (rec com temperatura 1,0, top-p 0,95 e top-k 20, os padrões do motor):

| Flint | chat (tok/s) | cópia (tok/s) | cópia: tokens / cobertura | leitura do prompt na cópia (tok/s) |
|---|---|---|---|---|
| **rec, especialistas INT3, bloco de KV de 16k** | **101,52** | 88,71 | 4.911 / 0,92 | 636 |
| llama.cpp UD-IQ3_XXS, sem drafter | 62,67 | 56,61 | 7.850 / 0,99 | 580 |
| llama.cpp UD-IQ3_XXS, só lookup | 62,26 | **121,78** | 6.972 / 0,98 | |
| llama.cpp UD-IQ3_XXS, DFlash2 3 drafts + lookup | 65,84 | 118,19 | 6.798 / 0,92 | |

- **No chat, o rec faz 1,5× o melhor llama.cpp** (101,5 contra 65,8 tok/s). Os 8 pedidos ficaram entre 99,9 e 102,6 tok/s.
- **Na cópia, o lookup do llama.cpp ainda ganha** (121,8 contra 88,7). O rec não especula, e o raciocínio do Flint repete
  o documento, que é justamente onde o n-gram acerta.
- **Os 125+ tok/s da Recursive** são de um M4 Pro de 48 GB com os especialistas em 4 bits. Aqui ficou em ~100. Os 3 bits
  quase não mudam o que cada token lê (~1,4 GB contra ~1,5 GB), porque as camadas densas continuam em 4 bits. A diferença
  não foi investigada.
- **Qualidade não foi medida em benchmark.** A requantização de 4 para 3 bits perde algo. A resposta de teste saiu
  correta, e a cobertura da cópia (0,92) é a mesma do llama.cpp com DFlash2. Antes de usar no dia a dia, vale um teste com as
  suas tarefas.
- **Outras quantizações não servem neste motor:** o kernel aceita só 3, 4 ou 8 bits, e só grupos de 64. Um GGUF como o
  UD-Q2_K_XL (13,6 GB) é formato do llama.cpp e não entra no rec.

## Por que llama.cpp e não MLX

O `Qwen3.8-27B-MLX-4bit` do LM Studio roda, mas o `mlx_lm` não faz especulação neste
modelo. Das 64 camadas, 48 são Gated DeltaNet, e o cache delas não volta atrás
quando um draft é rejeitado. Sem especulação, o teto é a banda de memória:
273 GB/s ÷ ~15 GB de pesos dá ~18 tok/s, e na prática menos. A velocidade do
HyperQwen vem justamente de verificar vários tokens por passada. O llama.cpp
**0.8.0 ou mais novo** (`brew upgrade llama.cpp`) faz isso com DFlash2, MTP e n-gram. A build 10360
só conhecia o DFlash v1 e recusa o GGUF do DFlash2 com `wrong number of tensors; expected 81, got 58`.

## E o MLX (mlx_vlm e oMLX)?

O `mlx_vlm` 0.7.10 especula neste modelo (`--draft-kind dflash|mtp`, com o drafter `dflash2` próprio) e
`mac/serve_mlx.sh` sobe essa configuração. **Neste Mac de 24 GB, ela só roda com a janela de carga do vigia.** O MLX abre o modelo sob demanda: o servidor
sobe com ~35% de memória livre, e o primeiro pedido materializa os pesos e empurra o resto do sistema para o swap.

| Tentativa (2026-10-30) | Pesos | Swap no primeiro pedido |
|---|---|---|
| MLX 4-bit do LM Studio (com visão) + DFlash2 MLX | 16,1 + 1,1 GB | +2,7 GB, servidor derrubado pelo vigia |
| MLX 4-bit só texto (`mac/mlx_text_only.py`) + MTP MLX | 14,1 GiB + 0,25 GB | +1,06 GB, servidor derrubado pelo vigia |

O llama.cpp não tem esse problema: ele usa o GGUF mapeado direto como memória da GPU e lê cada peso uma vez só.

**Com a janela de carga, o MLX coube, com e sem drafter.** O vigia ganhou uma janela de carga (`LOAD_GRACE_S=120 LOAD_GRACE_MB=2048`).
Na carga, o swap subiu entre 170 e 370 MiB (até ~1,1 GB com o DFlash2) e depois ficou parado. Durante o teste, a memória livre ficou
entre 12% e 26%. Mesmo protocolo da tabela de resultados, em 2026-10-30, `mlx_vlm` 0.7.10, alvo só texto 4-bit (14,1 GiB):

| Configuração | chat (tok/s) | aceitação | cópia (tok/s) | aceitação | prefill da cópia |
|---|---|---|---|---|---|
| MLX, sem drafter | 15,38 | — | 15,11 | — | 107,3 |
| MLX + MTP (0,25 GB) | **17,25** | 0,41 | 23,90 | 0,86 | 106,7 |
| MLX + DFlash2 (1,0 GB, quantização de terceiros) | 15,44 | 0,37 | 18,88 | 0,80 | 105,8 |
| mlx_vlm 0.7.11 (repetição): sem drafter / MTP / DFlash2 | 15,27 / 17,15 / 15,35 | | 15,05 / 23,18 / 19,22 | | ~106 |
| **oMLX 0.6.3rc1**, sem drafter (TurboQuant KV 4-bit) | 15,37 | — | 14,85 | — | 100,6* |
| oMLX + DFlash2, bloco 5, verificação `dflash` (a sua config do oMLX) | 17,55 | ~0,65** | 24,85 | | 97,6* |
| **oMLX + DFlash2, bloco 8, verificação `adaptive`** | **17,62** | | **27,53** | | 95,2* |
| llama.cpp sem especulação, para comparar | 10,07 | — | 10,28 | — | 95,8 |
| **llama.cpp DFlash2 + lookup (padrão)** | **18,05** | 0,31 | **38,28** | 0,96 | 90,0 |

\* O oMLX só devolve tempos em streaming. A leitura do prompt é medida no cliente (até o primeiro token) e inclui o HTTP.
\*\* Pelo log do oMLX: 64,5% num pedido de 512 tokens, 2,8 tokens por ciclo. A API dele não devolve a contagem de drafts.

Leitura:
- **O melhor MLX é o oMLX com DFlash2:** 17,6 tok/s no chat e 27,5 na cópia (bloco 8 adaptativo). O DFlash do oMLX
  (`dflash_mlx`) rende bem mais que o do `mlx_vlm` com o mesmo drafter (+15% no chat, +43% na cópia). Ele também carregou
  sem empurrar nada para o swap, e o swap até caiu durante a rodada.
- **Bloco 8 adaptativo contra bloco 5:** igual no chat em média. Ganha em inglês (20,1 e 23,6 contra 17,7 e 20,3) e perde em
  dinamarquês (13,9 contra 17,4), onde a aceitação baixa faz o bloco maior custar mais. Na cópia, ganha 11%.
- **A atualização do `mlx_vlm` (0.7.10 → 0.7.11) não mudou nada:** as diferenças ficaram em ±1%.
- **Sem especulação, o MLX é 53% mais rápido que o llama.cpp.** Por isso a especulação rende menos nele: verificar
  um bloco custa proporcionalmente mais quando a passada simples já é rápida.
- **No MLX, o MTP é o melhor drafter:** +12% no chat e +58% na cópia. O DFlash2 no MLX quase não ganha no chat (+0,4%).
  O `mlx_vlm` encolhe o bloco sozinho quando a aceitação cai, e o drafter é uma quantização de terceiros.
- **O melhor do Mac continua sendo o llama.cpp com DFlash2 + lookup:** 2,4% acima do oMLX no chat (18,05 contra 17,62) e
  1,4× na cópia (38,3 contra 27,5), por causa do lookup que nenhum dos dois servidores MLX tem. Ele também carrega sem janela de carga.

Comandos:

```bash
bash mac/download.sh mlx-text mlx-mtp     # alvo só texto (cópia local, sem download) + MTP
CHECK_ONLY=1 bash mac/serve_mlx.sh
bash mac/serve_mlx.sh                     # SPEC=mtp; SPEC=dflash com bash mac/download.sh mlx-dflash
LOAD_GRACE_S=120 SPEC=off bash mac/serve_mlx.sh   # o que coube aqui
python3 mac/bench.py mlx-off --model ~/models/mlx/Qwen3.8-27B-MLX-4bit-text
LOAD_GRACE_S=120 SPEC=mtp bash mac/serve_mlx.sh   # o melhor do mlx_vlm aqui
python3 mac/bench.py mlx-mtp --model ~/models/mlx/Qwen3.8-27B-MLX-4bit-text

# oMLX (o app em /Applications), numa pasta base própria; não toca em ~/.omlx
LOAD_GRACE_S=120 SPEC=dflash DFLASH_BLOCK=8 DFLASH_VERIFY=adaptive bash mac/serve_omlx.sh
python3 mac/bench.py omlx-dflash --model Qwen3.8-27B-MLX-4bit-text --stream
```

## Memória

Com 24 GB de memória unificada, a memória é a restrição principal. Por padrão, o macOS deixa a GPU fixar só
~2/3 da RAM (~16 GB), e só o modelo IQ4_XS já ocupa 14,25 GB.

| Peça | MiB |
|---|---|
| Qwen3.8-27B UD-IQ4_XS | ~13.590 |
| DFlash2 Q4_K_M (ou MTP Q4_0, ~1.300) | ~1.090 |
| KV q8_0, 32k tokens (16 camadas de atenção × 4 heads × 256) | ~1.090 |
| Estado GDN + 2 checkpoints | ~450 |
| Buffers de computação do Metal (alvo + drafter) | ~1.660 |
| Cache de prompt (`CACHE_RAM`) | 512 |
| **Total (`CTX=fast`)** | **~18.400** |

`CTX=long` (64k) soma mais ~1.090 MiB. O `serve.sh` calcula essa conta antes de carregar e recusa
subir quando:

- o LM Studio tem um modelo carregado (o script nunca descarrega nada por você);
- outro servidor local de modelo está rodando;
- o limite de memória fixada da GPU não cobre a conta;
- a memória livre não cobre a conta mais 1 GiB de margem.

Durante o uso, um vigia derruba o servidor se o swap crescer mais que 512 MiB
(`SWAP_LIMIT_MB`). Na primeira tentativa com MTP, o swap subiu 590 MiB e a velocidade caiu de 15 para 6 tok/s.

## Passo a passo

```bash
# 1. modelos (ficam em ~/.lmstudio/models, fora do Google Drive)
bash mac/download.sh

# 2. feche o LM Studio, o Chrome e outros apps pesados
#    (o serve.sh não chama `lms ps` se o LM Studio estiver fechado: esse comando acorda o serviço)

# 3. libere a GPU para fixar até 19 GiB (vale até reiniciar o Mac)
sudo sysctl iogpu.wired_limit_mb=19456

# 4. confira sem carregar nada
CHECK_ONLY=1 bash mac/serve.sh

# 5. sirva (API OpenAI em http://127.0.0.1:18020/v1, Web UI na mesma porta)
bash mac/serve.sh

# 6. em outro terminal, meça
python3 mac/bench.py m4pro-dflash
```

Para voltar o limite da GPU ao padrão sem reiniciar: `sudo sysctl iogpu.wired_limit_mb=0`.

## Configurações (`serve.sh`)

| Variável | Padrão | O que faz |
|---|---|---|
| `SPEC` | `dflash` | `dflash`, `mtp` ou `off` |
| `DRAFT_TOKENS` | 7 (dflash), 3 (mtp) | drafts por passada; o DFlash2 limita ao bloco treinado (8) |
| `LOOKUP` | `ngram-mod` | drafts do contexto: `ngram-mod`, `ngram-simple`, `ngram-map-k` ou `0` |
| `LOOKUP_MATCH` / `LOOKUP_MIN` / `LOOKUP_MAX` | 12 / 8 / 16 | tamanho do n-gram procurado, menor e maior draft do lookup |
| `CTX` | `fast` | `fast` = 32k, `long` = 64k, ou um número de tokens |
| `KV_TYPE` | `q8_0` | `f16`, `q8_0` ou `q4_0` |
| `NP` | 1 | slots do servidor (cada slot multiplica KV e estado) |
| `CACHE_RAM` | 512 | MiB do cache de prompt. Ele enche a cada pedido e com 2048 levou o MTP ao swap |
| `CTX_CHECKPOINTS` | 2 | snapshots do estado recorrente por slot, ~150 MB cada |
| `DRAFT_UBATCH` | 64 | batch do drafter; os logits de 248k tokens custam ~1 MiB por linha |
| `DRAFT_SAMPLING` | `probabilistic` | drafts sorteados em vez de argmax, como o `draft_sample_method` do HyperQwen |
| `PORT` / `BIND` | 18020 / 127.0.0.1 | igual ao HyperQwen; `VLLM_API_KEY` ou `api_key.txt` ativam `--api-key` |
| `WATCHDOG` / `SWAP_LIMIT_MB` | 1 / 512 | para o servidor se o swap crescer |
| `LOAD_GRACE_S` / `LOAD_GRACE_MB` | 0 / 2048 | janela depois da subida em que o swap pode crescer até `LOAD_GRACE_MB`; ao fechar, volta a valer `SWAP_LIMIT_MB` a partir dali |
| `EXTRA_ARGS` | | flags extras para o `llama-server` |

## Resultados

MacBook Pro M4 Pro (24 GB), llama.cpp 0.8.0 (build 11960) via Homebrew, `CTX=fast`,
KV q8_0, alvo UD-IQ4_XS. Medido em 2026-10-30 com `mac/bench.py`:

- **chat:** 4 prompts de `bench/prompts_real.jsonl`, 512 tokens cada, sampling padrão do modelo (T 1,0, top-p 0,95, top-k 20);
- **cópia:** reproduzir 8.000 caracteres (~2k tokens) de `docs/optimizations.md`.

| Configuração | chat (tok/s) | aceitação | cópia (tok/s) | aceitação | cobertura |
|---|---|---|---|---|---|
| `SPEC=off` (sem especulação) | 10,07 | — | 10,28 | — | 1,0 |
| `SPEC=dflash LOOKUP=0` | **18,38** (1,8×) | 0,27 | 28,74 (2,8×) | 0,57 | 1,0 |
| **`SPEC=dflash LOOKUP=ngram-mod` (padrão)** | **18,05** (1,8×) | 0,31 | **38,28** (3,7×) | 0,96 | 1,0 |
| `SPEC=mtp LOOKUP=ngram-mod` | 14,88 (1,5×) | 0,47 | 38,22 (3,7×) | 0,97 | 1,0 |

Leitura:
- **O DFlash2 é o melhor drafter no chat:** +21% sobre o MTP, como no HyperQwen em contexto curto. Ele aceita
  menos por draft (27-31% contra 47%), mas propõe 7 por passada em vez de 3.
- **O lookup não muda o chat e acelera a cópia em 33%.** Com ele, a cópia chega a 16,1 tokens por passada,
  que é o teto do `LOOKUP_MAX=16`. Subir esse teto é o próximo ajuste a medir.
- **Idioma pesa.** No prompt em dinamarquês, o DFlash2 caiu para 12,9 tok/s, com 15% de aceitação, contra 19-25 tok/s
  nos prompts em inglês (gotcha 61 do HyperQwen). Português ainda não foi medido.
- **O prefill é lento:** ~95 tok/s num prompt de 2k, ou seja, ~21 s antes do primeiro token. O decode é que manda no
  chat, mas para agentes com contexto longo isso pesa. `-ub`/`-b` ainda não foram ajustados.
- **A linha de base caiu ~6% na troca de versão:** 10,75 tok/s na build 10360 ([results/m4pro-iq4xs-b10360-off.json](results/m4pro-iq4xs-b10360-off.json)).
  A build 10360 não carrega o DFlash2, porque o GGUF tem 81 tensores e ela espera 58.
- **Memória:** depois de carregar, sobram 12-16% livres (~3-4 GB) com drafter e 24% sem drafter. Na primeira
  tentativa com MTP, o cache de prompt (2.048 MiB) e o ubatch do drafter (512) levaram o Mac ao swap
  (+590 MiB), e a velocidade caiu para 6 tok/s. Os padrões atuais (`CACHE_RAM=512`, `DRAFT_UBATCH=64`,
  `CTX_CHECKPOINTS=2`) rodaram as quatro configurações sem o swap crescer.

Os JSON de cada execução estão em [results/](results/), e o registro da sessão, em [`Atualizações/`](../Atualizações/README.md).
