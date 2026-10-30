# Qwen3.8-27B no MacBook M4 Pro 24 GB com as ideias do HyperQwen (llama.cpp + DFlash2)

**Timestamp:** 2026-10-30T13:09 (sessão iniciada por volta de 11:58)
**Branch:** `mac-apple-silicon` (criada a partir de `main` @ f382342 como `local-mac-mi50` e renomeada às ~15:45)

## Contexto / motivação

O usuário clonou o HyperQwen (syv-ai) e pediu duas coisas:

1. rodar o Qwen3.8-27B no MacBook M4 Pro de 24 GB, o mais rápido possível, usando as ideias do projeto;
2. depois, configurar o PC com 2× MI50 de 16 GB.

Esta sessão cobre o Mac. Às ~15:45, o usuário decidiu que a parte das MI50 não será implementada.

## Origem das informações e decisões

**Passado pelo usuário (2026-10-30):**
- O projeto HyperQwen e o objetivo (27B rápido no Mac, depois nas 2× MI50 16 GB).
- Os modelos que já estavam no LM Studio, entre eles o `incoai/Qwen3.8-27B-DFlash2-GGUF`, baixado em 19/08.
- Uma análise da viabilidade do HyperQwen nas MI50 (gfx906), colada no meio da sessão, com o fork `vllm-gfx906-mobydick`,
  os patches portáveis e os números esperados. **Quem escreveu essa análise está por confirmar** (veio de fora desta sessão),
  e as afirmações dela ainda não foram verificadas.

**Escolhas do usuário nas perguntas da sessão:**
- branch nova;
- alvo UD-IQ4_XS (14,25 GB);
- trocar o DFlash2 pela revisão com a correção de RoPE;
- baixar o MTP;
- `brew upgrade llama.cpp`;
- rodar as 4 configurações;
- deixar o LM Studio fechado até o fim dos testes.

**Proposto pelo Claude (com evidência nesta sessão):**
- Usar o llama.cpp no Metal em vez do MLX, porque o `mlx_lm` não especula em modelo híbrido.
- O mapeamento das ideias do HyperQwen para flags do llama.cpp.
- Os padrões de memória do `serve.sh`.

## Diagnóstico

- O HyperQwen é vLLM + CUDA (Marlin, FA2, pth-flash-attn, FlashInfer) e não roda no Metal.
- O `Qwen3.8-27B-MLX-4bit` do LM Studio roda no `mlx_lm` 0.31.3, mas sem especulação: o cache das 48 camadas Gated DeltaNet
  não volta atrás quando um draft é rejeitado.
- O llama.cpp do Homebrew (build 10360) tinha `--spec-type draft-dflash,draft-mtp,ngram-*`, mas só o DFlash v1. O GGUF do DFlash2
  falhou com `done_getting_tensors: wrong number of tensors; expected 81, got 58`, inclusive na conversão do ggml-org.
  A versão 0.8.0 (build 11960) carrega.
- O DFlash2 local (18/08, sha256 `18a380ef…`) era a revisão de lançamento, sem `dflash.rope.dimension_sections`, a correção de mrope
  publicada em 29/08. Foi trocado pela revisão `144eb06` (sha256 `d169ad33…`).
- Não havia GGUF do Qwen3.8-27B base. O `LBS-27B-IQ4_XS` local é um fine-tune para leitura de telas e interfaces (tags: gui-grounding).
- A pasta `incoai/Qwen3.8-27B-DFlash2` (safetensors) está vazia: o download nunca terminou.
- Memória: por padrão, o macOS deixa a GPU fixar ~2/3 da RAM. O `iogpu.wired_limit_mb` já estava em 20480 nesta máquina
  (vale até reiniciar).

## Alterações realizadas

- `mac/serve.sh` (novo): launcher do `llama-server`.
  - Knobs no estilo do HyperQwen: `SPEC`, `LOOKUP`, `CTX`, `DRAFT_TOKENS`, `KV_TYPE`, `NP`, `CACHE_RAM`, `CTX_CHECKPOINTS`,
    `DRAFT_UBATCH`, `DRAFT_SAMPLING`, `PORT=18020`, `BIND=127.0.0.1` e a chave de API via `resolve_api_key.sh`.
  - Preflight com a conta de memória. Recusa subir quando:
    - o LM Studio tem modelo carregado;
    - outro servidor de modelo está rodando;
    - o limite da GPU é baixo;
    - falta memória livre.
  - `CHECK_ONLY=1` faz só a conta. Um vigia derruba o servidor se o swap crescer mais que 512 MiB.
- `mac/bench.py` (novo): benchmark de um stream, só com a biblioteca padrão. Usa os `timings` do servidor para decode, prefill e
  aceitação. Dois cenários: chat (`bench/prompts_real.jsonl`) e cópia de documento, com cobertura por `bench/verbatim.py`.
- `mac/download.sh` (novo): baixa o alvo, o DFlash2 e o MTP para `~/.lmstudio/models`, fora do Google Drive, e avisa se o DFlash2
  não tem a correção de RoPE.
- `mac/README.md` (novo): mapeamento das ideias, conta de memória, passo a passo, knobs e resultados.
- `README.md`: uma linha na tabela "Running it" apontando para `mac/`.
- Fora do repositório:
  - `brew upgrade llama.cpp` (10360 → 0.8.0);
  - downloads em `~/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/` (UD-IQ4_XS e `MTP/mtp-Qwen3.8-27B-Q4_0.gguf`);
  - DFlash2 substituído em `~/.lmstudio/models/incoai/Qwen3.8-27B-DFlash2-GGUF/`.

## Resultados

llama.cpp 0.8.0, `CTX=fast` (32k), KV q8_0, UD-IQ4_XS, sampling padrão (T 1,0). Os 4 prompts de chat têm 512 tokens cada.
O teste de cópia reproduz ~2k tokens.

| Configuração | chat tok/s | aceitação | cópia tok/s | aceitação | cobertura |
|---|---|---|---|---|---|
| sem especulação | 10,07 | — | 10,28 | — | 1,0 |
| DFlash2 | 18,38 | 0,27 | 28,74 | 0,57 | 1,0 |
| **DFlash2 + lookup ngram-mod (padrão)** | **18,05** | 0,31 | **38,28** | 0,96 | 1,0 |
| MTP + lookup | 14,88 | 0,47 | 38,22 | 0,97 | 1,0 |

- Na build 10360, a linha de base fez 10,75 tok/s no chat. O DFlash2 não carregava nessa versão.
- O prefill ficou em ~95 tok/s com um prompt de 2k.
- O prompt em dinamarquês derruba a aceitação do DFlash2: 15%, 12,9 tok/s.
- Com o lookup, a cópia aceita 16,1 tokens por passada, que é o teto do `LOOKUP_MAX=16`.
- O swap não cresceu nas quatro configurações da 0.8.0.

## Incidentes

- **Swap na primeira rodada com MTP (build 10360).** Com `CACHE_RAM=2048` e o ubatch do drafter em 512, a memória livre caiu para 7%,
  o swap subiu ~590 MiB e o decode ficou em 6 tok/s. Parei o servidor na hora e baixei os padrões (`CACHE_RAM=512`,
  `CTX_CHECKPOINTS=2`, `DRAFT_UBATCH=64`). A conta de memória passou a incluir o cache de prompt.
- **O preflight acordou o LM Studio.** `lms ps` sobe o serviço do LM Studio se ele estiver fechado. Isso aconteceu depois de o usuário
  ter fechado o app. A tentativa de fechá-lo foi bloqueada pelo classificador de segurança, e o usuário fechou por conta própria.
  O `serve.sh` agora só chama `lms ps` se o LM Studio já estiver rodando.

## Teste com mlx_vlm (pedido do usuário, ~13:15)

- O `mlx_vlm` 0.7.10 especula neste modelo híbrido (`--draft-kind dflash|mtp`, drafter `dflash2` próprio) e devolve
  `timings` com os mesmos campos do llama.cpp. O `mlx_lm` não especula.
- Novos arquivos:
  - `mac/serve_mlx.sh`, com preflight e vigia de swap;
  - `mac/common.sh`, com o preflight e o vigia agora compartilhados com o `serve.sh`;
  - `mac/download.sh mlx-dflash mlx-mtp`, que baixa para `~/models/mlx-drafters`:
    - `SSSAMMMM/Qwen3.8-27B-DFlash2-MLX-4bit` (1,08 GB, exportação 4-bit de terceiros do z-lab DFlash2);
    - `mlx-community/Qwen3.8-27B-MTP-4bit` (0,24 GB).
- `mac/bench.py`: o modelo agora é explícito, porque o `mlx_vlm` carrega ou baixa o modelo que o pedido nomear. `/props` ficou opcional.
- Corrigido: o aviso falso de "DFlash2 sem mrope" no `download.sh`. Com `set -o pipefail`, o `grep -q` fecha o pipe e o teste falhava.
- **Incidente:** com o alvo `lmstudio-community/Qwen3.8-27B-MLX-4bit` (16,1 GB, 879 MiB de visão) mais o DFlash2 MLX:
  - o servidor subiu com 35% livre (o MLX carrega os pesos sob demanda);
  - o primeiro pedido (55 tokens) materializou os pesos e o swap cresceu 2,7 GB;
  - o vigia do `serve_mlx.sh` derrubou o servidor sozinho, e a sequência foi interrompida;
  - o macOS criou mais um arquivo de swap (2 → 3 GB).
  O prefill de 55 tokens levou 8,3 s, e o decode do aquecimento (9,5–14,8 tok/s), sob swap, não serve de comparação.
- Conclusão: neste Mac de 24 GB, o MLX 4-bit com visão mais o DFlash2 não cabe sem swap. O llama.cpp com IQ4_XS cabe.
- **Segunda tentativa (escolha do usuário: "MLX só texto + MTP"):**
  - `mac/mlx_text_only.py` (novo) copia por faixa de bytes só os tensores de texto (14.432 MiB, contra 15.336) para
    `~/models/mlx/Qwen3.8-27B-MLX-4bit-text`, com `model_type: qwen3_5_text`. A carga estrutural (`lazy=True`, estrita) passou.
  - Dois tropeços antes de carregar, nenhum com pesos na memória:
    1. o `mlx_vlm` exige um processador;
    2. o `tokenizer_config.json` declara `processor_class: Qwen3VLProcessor`.
    O script agora tira essa chave e não copia os configs de imagem, e o `AutoProcessor` cai no tokenizer.
  - Na carga real, o servidor subiu com 35% livre. O primeiro pedido (55 tokens) fez o swap crescer 1.060 MiB, e o vigia
    (`SWAP_LIMIT_MB=256`) derrubou o servidor. O decode do aquecimento, sob swap, deu 18,7 tok/s em 27 tokens e não serve de comparação.
  - `mac/serve_mlx.sh` passou a usar o alvo só texto e `SPEC=mtp` por padrão, com `PREFILL_STEP=512` e vigia em 256 MiB.
    `mac/download.sh mlx-text` gera o alvo.
  - Conclusão: pela regra de parar ao subir o swap, o `mlx_vlm` não é viável neste Mac de 24 GB. O caminho continua sendo o llama.cpp.
    Os scripts MLX ficam documentados em `mac/README.md` para máquinas com mais memória.
- **Terceira tentativa (pedido do usuário: "mlx sem mtp nem dflash", ~13:38):**
  - O vigia de `mac/common.sh` ganhou uma janela de carga opcional (`LOAD_GRACE_S`/`LOAD_GRACE_MB`). Durante a janela, o swap pode
    crescer até `LOAD_GRACE_MB`. Ao fechar, a referência é relida e `SWAP_LIMIT_MB` volta a valer.
  - O usuário escolheu a janela de 120 s com teto de 2 GB, sabendo que isso afrouxa a regra de parar quando o swap sobe.
  - Resultado: na carga, o swap subiu 174 MiB e não saiu mais nada para o swap (swapouts parados). A memória livre ficou em 26%
    durante o teste.
  - MLX só texto, sem drafter: chat **15,38 tok/s**, cópia 15,11 tok/s (cobertura 1,0), prefill 89,7 e 107,3 tok/s
    (`mac/results/m4pro-mlx4bit-text-off.json`). Fica 53% acima do llama.cpp sem especulação (10,07) e 15% abaixo do
    llama.cpp com DFlash2 + lookup (18,05; 38,28 na cópia).
- **Quarta rodada (pedido do usuário: "teste com MTP depois com DFlash2", ~13:48)**, com a mesma janela de carga:
  - MLX + MTP: chat 17,25 tok/s (aceitação 0,41) e cópia 23,90 tok/s (0,86).
  - MLX + DFlash2: chat 15,44 tok/s (0,37) e cópia 18,88 tok/s (0,80).
  - Swap na carga: +242 e +172 MiB, segundo o vigia de cada servidor. O monitor geral chegou a ver +1,1 GB na troca para o DFlash2,
    e depois disso não houve mais swapouts.
  - Resultados em `mac/results/m4pro-mlx4bit-text-{mtp,dflash}.json`.
  - Conclusão:
    - o melhor do Mac continua sendo o llama.cpp com DFlash2 + lookup (18,05 / 38,28);
    - o MLX + MTP fica 4% abaixo no chat e bem abaixo na cópia, porque não tem lookup;
    - sem especulação, o MLX é mais rápido (15,4 contra 10,1).
- Rechecagem de 1 h, durante o teste: o usuário escolheu não ativar o LLM local.

## Atualização do mlx-vlm e teste do oMLX (pedido do usuário, ~14:25)

- **Versões:**
  - O usuário perguntou se os modelos e o MLX eram os mais atuais. Todos os modelos estavam na última revisão; os arquivos MLX
    batem em SHA-256 com o Hugging Face. O mlx 0.32.4 também era o último. Não há Qwen mais novo que caiba: o Qwen3.8-Flash-Next
    tem 180B.
  - `pip install -U mlx-vlm`: 0.7.10 → 0.7.11.
  - O `mlx-lm` 0.32 exige Python ≥ 3.11, e o pyenv global é 3.10.14, então ficou na 0.31.3. Ele não é usado nos testes.
- **mlx_vlm 0.7.11:** sem drafter 15,27 / 15,05; MTP 17,15 / 23,18; DFlash2 15,35 / 19,22 (chat / cópia, tok/s). Diferenças de ±1%
  em relação à 0.7.10.
- **oMLX 0.6.3rc1** (`/Applications/oMLX.app`, CLI `omlx`, Python 3.11 próprio com mlx 0.32.0, mlx-lm 0.31.3, mlx-vlm 0.6.3
  e `dflash_mlx` 0.1.10):
  - Novo `mac/serve_omlx.sh`. Ele usa uma pasta base própria (`~/models/omlx-bench`) e não toca em `~/.omlx`. A pasta de modelos
    (`~/models/omlx-models`) tem links para os pesos só texto e um `config.json` `qwen3_5` sem `vision_config`, que o oMLX serve
    pelo mlx-lm.
  - Configurações herdadas do usuário (`~/.omlx/model_settings.json`): TurboQuant KV 4-bit; DFlash com bloco 5 e verificação
    `dflash`. O drafter configurado lá apontava para a pasta vazia `incoai/Qwen3.8-27B-DFlash2`, então aquela configuração não
    funcionaria. Aqui foi usado o DFlash2 MLX 4-bit já quantizado. Sampling com top-k 20.
  - `mac/bench.py --stream` (novo): o oMLX só devolve tempos (`usage.generation_tokens_per_second`) no último pedaço de um
    stream. A primeira rodada saiu zerada (0,00 tok/s) e foi parada e repetida.
  - O servidor troca o nome do processo para `omlx-server`. O `pkill` antigo não o pegou, e o modelo ficou carregado até ser
    encerrado pelo PID. Os padrões de `mac/common.sh` e do script de testes foram corrigidos.
  - Resultados (chat / cópia, tok/s):
    - sem drafter: 15,37 / 14,85;
    - DFlash2 bloco 5 `dflash`: 17,55 / 24,85 (aceitação de ~64,5% no log);
    - DFlash2 bloco 8 `adaptive`: 17,62 / 27,53.
  - Swap: caiu 279 MiB ao longo da rodada. Nenhuma das três cargas empurrou nada para o swap.
- **Conclusão:**
  - o melhor MLX deste Mac é o oMLX com DFlash2, bloco 8 adaptativo;
  - o melhor do Mac continua sendo o llama.cpp com DFlash2 + lookup: +2,4% no chat e 1,4× na cópia.

## Mais de 60 tok/s: Recursive Flint (pedido do usuário, ~15:10)

- **Origem:**
  - O usuário passou a página https://recursive.com/updates/introducing-recursive-flint ("o fabricante promete mais de 60 t/s")
    e pediu para ajustar a configuração até esse patamar.
  - Depois, o usuário lembrou que eles usam o motor `rec`.
- **Diagnóstico:**
  - A página é de outro modelo, o Recursive Flint (MoE 40B no total, 2,5B ativos), e não do Qwen3.8-27B. Os 125+ tok/s
    foram medidos num M4 Pro de 48 GB com o `rec`.
  - O Qwen3.8-27B é denso, com teto de ~19 tok/s sem especulação neste Mac. 60 tok/s em conversa não é alcançável com ele.
  - O `rec` (recursive-oss/rec v0.1.1) exige 48 GiB por bloqueio rígido no CLI e no manifesto (`min_ram_bytes`), e o modelo dele
    (`flint-int4` MLX) tem 22,6 GB. Não roda neste Mac de 24 GB.
- **Verificação:** o usuário disse que já havia um Flint baixado no LM Studio. A busca não encontrou nenhum:
  - nem por nome nem pela arquitetura (`flint` nos GGUF e nos `config.json`);
  - nem em downloads parciais ou no cache do Hugging Face.
  Foram encontradas pastas vazias de `Qwen3-Coder-30B-A3B`, `gemma-3n` e `medgemma`.
- **Download (com aval):** `unsloth/flint-GGUF` UD-IQ3_XXS (14,54 GB) para `~/.lmstudio/models/unsloth/flint-GGUF/`.
  O llama.cpp 0.8.0 já conhece a arquitetura `flint`.
- **`mac/serve.sh`:** novo `MODEL_PRESET=flint`, com a geometria de KV do Flint (14 de 49 camadas com atenção completa, 4
  cabeças KV, dimensão 128), `SPEC=off` e `CTX=long` por padrão, e alias `flint`. `mac/download.sh flint`.
- **`mac/bench.py`:** `--quote-max-tokens`, e a cópia passou a registrar `answer_chars`, `reasoning_chars` e `finish_reason`.
- **Resultados** (chat / cópia, tok/s):
  - sem lookup: 62,67 / 56,61 (cópia com 8.000 tokens: cobertura 0,99, 7.850 tokens);
  - com lookup: 62,26 / 121,78 (aceitação de 0,95, cobertura 0,98, 6.972 tokens).
  - Prefill do chat: 250–313 tok/s.
  - Com o limite antigo de 2.400 tokens, a cópia dava cobertura 0: o Flint sempre pensa, e o template não tem chave para desligar.
- Memória: ~16,6 GB estimados, 16–19% livres com o modelo carregado e nenhum crescimento de swap.
- Qualidade do Flint em IQ3_XXS: não medida.

## Flint com DFlash2 (pedido do usuário: "mais otimizações com MTP ou DFlash2", ~16:05)

- O Flint não tem MTP. Existe o `incoai/Flint-DFlash2-GGUF` (26/10, bloco 8, alvo `RecursiveSI/flint`). O cartão dele indica
  aceitação de 3,5–4,5 tokens por passada e 1,8–2,3× de velocidade numa GB300 em BF16. Baixei o Q8_0 (0,43 GB).
- `mac/serve.sh`: o preset `flint` passou a apontar para esse drafter e recusa `SPEC=mtp`. `mac/download.sh flint-dflash`.
- Duas rodadas, com aval, com o LM Studio fechado pelo usuário antes da primeira. DFlash2 + lookup (chat / cópia, tok/s):
  - 7 drafts: 54,86 / 105,94;
  - 5 drafts: 62,45 / 111,21;
  - **3 drafts: 65,84 / 118,19**;
  - 4 drafts: 63,17 / 106,85;
  - 2 drafts: 60,03 / 109,77;
  - 3 drafts com sorteio greedy: 60,07 / 104,50.
  
  Aceitação no chat: 0,25 (7 drafts) a 0,48 (2 drafts). Nenhum crescimento de swap.
- Conclusão:
  - No Mac, o DFlash2 rende pouco num MoE: verificar mais tokens acorda mais especialistas, e a aceitação com o alvo IQ3 fica
    abaixo da do cartão.
  - O melhor (3 drafts, +5% no chat) está perto da variação entre execuções.
  - Ficou como padrão do preset `flint` (`SPEC=dflash DRAFT_TOKENS=3`), como combinado com o usuário. `SPEC=off` volta ao Flint
    sem drafter.
  - Na cópia, o melhor continua sendo o lookup sozinho (121,8).
- A cópia não terminou dentro dos 8.000 tokens em 4 das 6 configurações, porque o raciocínio variou entre ~18 e ~30 mil caracteres.

## Teste do rec (pedido do usuário, ~16:30)

- O instalador oficial foi lido antes (binário com SHA-256 em `~/.local/bin`, dados em `~/.local/share/rec`, linha de PATH no
  `~/.zshrc`). O classificador de segurança bloqueou a instalação feita por mim, então o usuário rodou a instalação com
  `SHELL=/bin/sh`, para não alterar o `~/.zshrc`.
- Resultado: rec 0.1.1, modelo `flint (flint-int4)`, motor `mlx (metal)`. A instalação parou com "This Mac does not meet the
  requirements / rec: this model requires 48 GiB of memory; detected 24 GiB". O binário não foi instalado, o `~/.zshrc` não
  mudou e ficaram só pastas vazias em `~/.local/share/rec`.
- Conclusão: o rec não roda em Macs de 24 GB. O caminho do Flint aqui é o llama.cpp (`MODEL_PRESET=flint`).

## Forks do rec e Tempo9 (pedido do usuário, ~16:45)

- **Forks:** os 43 forks do `recursive-oss/rec` foram comparados branch por branch com o original. Para 24 GB, só
  `thinkspread/rec@local/flint-e3-24gb` serve. Ele remove a trava de 48 GiB, aceita `routed_expert_bits` 3 e chama
  `mx.set_wired_limit`, mas depende de um checkpoint `flint-int4-e3` não publicado. `ericzhoun/rec@apple-m5-32gb` só remove a trava.
  Os outros tratam de outros chips (M1/M2/M3/M4 Max, ROCm, CUDA) ou de funções do servidor.
- **Tempo9 (ThinkSpread, motor fechado, Apache-2.0 no SDK e no CLI):** o usuário perguntou se o thinkspread.com é confiável (o site
  não respondia, nem para mim).
  - Avaliação passada a ele: desenvolvedor identificável (Jiejing Zhang, GitHub desde 2009, 29 commits no `modelscope/dash-infer`);
    binário assinado com Developer ID e notarizado; SHA-256 conferido com a fórmula; domínio e organização criados em 17/08/2026;
    0 estrelas.
  - O usuário decidiu seguir.
- **Downloads e instalação:**
  - `thinkspread/Flint-Q6K-IQ3XXS-GGUF` (18,03 GB), com SHA-256 igual ao publicado;
  - `brew install thinkspread/tap/tempo9`, versão 1.2.1;
  - `mac/serve_tempo9.sh` (novo) e `mac/download.sh flint-tempo9`;
  - `mac/bench.py`: streaming de JSON puro, contagem de pedaços quando não há `usage`, recurso ao relógio de parede quando
    nada visível é transmitido, e `e2e_tok_s`.
- **Resultados no M4 Pro:**
  - Tempo9 na CPU (o Metal fica desligado em chips anteriores ao M5, por projeto): ~16–17 tok/s;
  - `AS_METAL_TENSOR_FORCE=1`: o Metal sobe, mas a resposta a "17*23" foi lixo, 3.000 tokens a ~67 tok/s, com 7% de memória livre;
  - o mesmo GGUF no llama.cpp deixou 8% livres depois de carregar e foi cancelado pela trava de 12%.
  
  Em nenhum dos testes o swap cresceu de forma sustentada, e o `tempo9` só abriu conexão em `127.0.0.1:18020`.
- **Conclusão:** o Tempo9 só vale em Mac com M5. Para o M4 Pro de 24 GB, o Flint continua no llama.cpp com o UD-IQ3_XXS.
- **Limpeza (aprovada pelo usuário):** removidos o `tempo9` (`brew uninstall`), `~/.lmstudio/models/thinkspread` (17 GB),
  `~/.cache/tempo9`, `~/models/mlx` (cópia MLX só texto, 14 GB), `~/models/omlx-models` e `~/models/omlx-bench`. O disco foi de
  13 para 45 GB livres. O tap `thinkspread/tap` continua no Homebrew. O `serve_mlx.sh` e o `serve_omlx.sh` voltam a funcionar
  depois de `bash mac/download.sh mlx-text`.

## Issues remanescentes

- Ajustes ainda não medidos (cada um precisa de aval para carregar o 27B):
  - `LOOKUP_MAX` 24 ou 32 na cópia;
  - `DRAFT_TOKENS` 4 ou 5 no DFlash2;
  - prompts em português;
  - `-ub`/`-b` para o prefill;
  - comparação em modo greedy.
- O vocabulário reduzido do drafter não tem caminho no llama.cpp para qwen35 (PR #29143 em aberto).
- O LM Studio (LLM local dos resumos do ai-memory) ficou fechado a pedido do usuário. Precisa ser religado depois.
- MI50: fora do escopo, por decisão do usuário (~15:45). A análise colada no meio da sessão fica só como registro.
- Publicação (pedido do usuário, ~15:35): commit e push para o fork público `leolimalab/HyperQwen`, do `syv-ai/HyperQwen`.
  A visibilidade foi escolha do usuário. Os JSON em `mac/results/` levam caminhos locais (`/Users/...`).
- Às ~15:45, a branch foi renomeada de `local-mac-mi50` para `mac-apple-silicon` (no GitHub e localmente), porque só o Mac foi implementado.

## Arquivos tocados

- `mac/serve.sh`, `mac/serve_mlx.sh`, `mac/serve_omlx.sh`, `mac/common.sh`, `mac/mlx_text_only.py`, `mac/bench.py`, `mac/download.sh`, `mac/README.md` (novos)
- Fora do repositório:
  - `~/.lmstudio/models/unsloth/flint-GGUF/flint-UD-IQ3_XXS.gguf` (14,5 GB);
  - `~/.lmstudio/models/incoai/Flint-DFlash2-GGUF/Flint-DFlash2-Q8_0.gguf` (0,43 GB);
  - `~/.lmstudio/models/thinkspread/Flint-Q6K-IQ3XXS-GGUF/` (18 GB) e o `tempo9` 1.2.1 via Homebrew (removidos depois do teste);
  - `~/models/mlx-drafters/` (DFlash2 e MTP em MLX);
  - `~/models/mlx/Qwen3.8-27B-MLX-4bit-text/` (14,1 GiB);
  - `~/models/omlx-models/` (só links);
  - `~/models/omlx-bench/` (configuração e cache do oMLX de teste);
  - `pip install -U mlx-vlm` (0.7.11) no Python global do pyenv.
- `mac/results/*.json` (5 execuções)
- `README.md` (uma linha)
- `Atualizações/README.md`, `Atualizações/2026-10-30-mac-m4pro-llamacpp-dflash2.md` (novos)
