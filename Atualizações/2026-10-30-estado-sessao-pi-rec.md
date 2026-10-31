# Estado da sessão: rec com o Flint INT3 e o Pi (para retomar)

**Timestamp:** 2026-10-30T23:36
**Pedido do usuário:** "salve o estado dessa sessao para continuar depois"
**Registros desta sessão:**
- [2026-10-30-rec-flint-int3-m4pro.md](2026-10-30-rec-flint-int3-m4pro.md): motor do rec, checkpoint INT3, resultados e Pi;
- [2026-10-30-estado-sessao-e-proximos-passos.md](2026-10-30-estado-sessao-e-proximos-passos.md): estado anterior, com o llama.cpp e o MLX.

## Onde as coisas estão

- **Repositório:** branch `mac-apple-silicon` em `af61fbe`, enviada ao fork (`leolimalab`).
  - A `main` do fork (`a3dc382`) não tem `30e7638`, `125c075` nem `af61fbe`. Falta um PR ou um merge.
- **Sem commit:**
  - `mac/pi_rec.sh` (novo);
  - a seção "Pi (pi.dev) com o rec" em `2026-10-30-rec-flint-int3-m4pro.md`;
  - o comentário do `mac/serve_rec.sh`, que dizia "cache do alocador em 512 MiB" quando o patch o desliga (corrigido agora);
  - este arquivo e a linha dele no `README.md`.
- **Mac no momento:**
  - nenhum servidor de modelo rodando;
  - LM Studio fechado pelo usuário;
  - ai-memory ativo (405);
  - `iogpu.wired_limit_mb=20480`, que volta ao padrão no reboot;
  - 27 GiB livres no disco.
- **Fora do repositório:**

  | Caminho | O que é |
  |---|---|
  | `~/models/flint-int4-e3` | checkpoint com especialistas INT3 (17,0 GiB), manifesto `flint-int4-e3.json` validado |
  | `~/models/flint-int4` | só os arquivos pequenos do original; os shards foram apagados na conversão |
  | `~/src/rec-e3` | fork `thinkspread/rec@local/flint-e3-24gb` (325d244) com `mac/rec-e3-kv16k.patch` aplicado |
  | `~/.venvs/rec-e3` | Python 3.12, mlx 0.32.0, rec-mlx com o patch |
  | `~/.local/share/hyperqwen-pi` | ainda não existe: o Pi não foi instalado |

## Resultados medidos (M4 Pro 24 GB, um stream, tok/s)

| Flint | chat | cópia |
|---|---|---|
| **rec, especialistas INT3, bloco de KV de 16k, cache 0** | **101,52** | 88,71 |
| llama.cpp UD-IQ3_XXS, DFlash2 3 drafts + lookup | 65,84 | 118,19 |
| llama.cpp UD-IQ3_XXS, só lookup | 62,26 | 121,78 |

O rec chega a 19,85 GB de pico, sem swap, e só roda com LM Studio, Chrome, RustDesk e Monitor de Atividade fechados.

## Como retomar: teste do Pi com o rec

1. **Instalar o Pi (usuário).** O classificador do modo automático bloqueou o `npm ci` rodado pelo Claude, e isso não deve ser
   contornado:
   ```
   ! INSTALL_ONLY=1 bash mac/pi_rec.sh
   ```
2. **Pedir aval na hora** (AskUserQuestion): o servidor usa ~19–20 GB, e LM Studio, Chrome, RustDesk e Monitor de Atividade
   precisam estar fechados. O aval de uma sessão anterior não vale.
3. **Subir o servidor com o vigia:**
   ```
   SWAP_LIMIT_MB=512 LOAD_GRACE_S=60 LOAD_GRACE_MB=1024 bash mac/serve_rec.sh
   ```
   Num monitor, montar o padrão do processo em partes (`B="bin/rec"; PAT="${B}-mlx serve"`), para ele não casar com a própria
   linha de comando.
4. **Abrir o Pi numa pasta de teste vazia**, porque as ferramentas dele leem, escrevem e rodam comandos na pasta:
   ```
   mkdir -p ~/pi-teste && cd ~/pi-teste && bash "<repo>/mac/pi_rec.sh"
   ```
   Sem interface: `bash "<repo>/mac/pi_rec.sh" -p "liste os arquivos"`.
5. **O que conferir:**
   - o Pi conecta ao provider `rec` e ao modelo `flint`;
   - as chamadas de ferramenta funcionam (o servidor usa a gramática `flint`);
   - o prompt de sistema e as ferramentas do Pi cabem nos 16k de contexto;
   - o swap fica em 0 e a velocidade perto de 100 tok/s;
   - se o servidor recusar pedidos por causa do `max_tokens` que a extensão manda, baixar `maxTokens` no `REC_PI_CONFIG` do
     `pi_rec.sh`.

## Regras do usuário que valem aqui

- Modelo pesado só com AskUserQuestion naquele momento, dizendo os GB e o que sai da memória. Parar na hora (TaskStop) se o swap
  subir ou a velocidade cair.
- Nunca chamar `lms ps` com o LM Studio fechado. Nunca descarregar modelos do LM Studio.
- Commit e push só quando o usuário pedir. Registrar tudo em `Atualizações/`.
- venv e `node_modules` fora do Drive.

## Pendências (nenhuma pedida ainda)

- Commit de `mac/pi_rec.sh` e dos registros, quando o usuário pedir.
- PR ou merge da branch para a `main` do fork.
- Teste de qualidade do INT3 contra o llama.cpp IQ3_XXS (GSM8K ou tarefas do usuário).
- Teste do patch de KV com contexto acima de 16.384 tokens (`CTX=32768`, que soma 0,45 GiB).
- Citar o `mac/pi_rec.sh` no `mac/README.md` depois do teste.
- O LM Studio apareceu aberto no meio da sessão, mais de uma hora depois de o usuário recusar ativá-lo. Origem a confirmar.
