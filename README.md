## Parte 1: estrutura do projeto

### Visão geral das pastas

- `DATA/`: dados usados pelo projeto.
- `OUTPUTS/`: saídas de treino, avaliação, checkpoints, logs e arquivos de submissão.
- `conf/`: configurações Hydra do projeto.
- `datasets/`: implementação dos datasets e leitores de dados.
- `models/`: definição dos modelos.
- `func/`: loop de treino, avaliação e geração de submissão.
- `common/`: utilitários gerais, logging, scheduler, transforms e helpers distribuídos.
- `loss_fn/`: funções de perda.
- `expts/`: arquivos `.txt` com overrides de experimento.
- `docs/`: documentação auxiliar sobre datasets e modelos.
- `external/`: dependências externas e anotações auxiliares.
- `sample_scripts/`: scripts utilitários auxiliares.
- `notebooks/`: funções de análise e fusão de resultados.

### O que existe dentro de `DATA/`

O código espera os dados principalmente nestes diretórios:

- `DATA/videos/`: vídeos brutos.
- `DATA/external/`: features externas, principalmente do RULSTM.
- `DATA/extracted_features/`: features extraídas previamente por outros backbones.
- `DATA/pretrained/`: pesos pré-treinados, por exemplo checkpoints do TIMM.

Para EPIC-Kitchens, os caminhos-base usados nas configs estão em:

- [conf/dataset/epic_kitchens/common.yaml](/media/storage/alana/AVT/conf/dataset/epic_kitchens/common.yaml:1)
- [conf/dataset/epic_kitchens100/common.yaml](/media/storage/alana/AVT/conf/dataset/epic_kitchens100/common.yaml:1)

Hoje esses arquivos apontam, por padrão, para:

- `DATA/videos/EpicKitchens/videos_ht256px`
- `DATA/videos/EpicKitchens100/videos_extension_ht256px`
- `DATA/external/rulstm/RULSTM/data_full/`
- `DATA/external/rulstm/RULSTM/ek100_data_full/`

Se seus dados estiverem em outro lugar, ajuste essas configs.

### O que existe dentro de `OUTPUTS/`

Cada experimento roda em uma pasta derivada do arquivo em `expts/`.

Exemplo:

- config: `expts/01_ek100_avt.txt`
- saída: `OUTPUTS/expts/01_ek100_avt.txt/`

Dentro dessa pasta, normalmente aparecem:

- `0/`, `1/`, `2/`...: runs numerados de sweep/multirun.
- `local/`: saída usada no modo debug local (`-g`).
- `.submitit/`: metadados de execução via Submitit/SLURM.
- `multirun.yaml`: arquivo Hydra do multirun.

Dentro de uma pasta de run, por exemplo `OUTPUTS/expts/01_ek100_avt.txt/0/`, aparecem:

- `checkpoint.pth`: último checkpoint salvo do treino.
- `checkpoint_best.pth`: melhor checkpoint, apenas se `train.store_best=true`.
- `AVT.log`: log principal.
- `.hydra/`: configuração Hydra daquela execução.
- `.hydra.orig/`: backup da configuração Hydra.
- `results/`: predições e resultados da avaliação.
- `results_test/`: resultados de inferência no split de teste, quando a submissão é gerada.
- `challenge/`: onde o código salva `test.json` e `submit.zip`.
- `wandb/`: metadados do Weights & Biases.
- `logs/`: logs auxiliares, quando presentes.

### Onde ficam os checkpoints

Os checkpoints ficam dentro da pasta da run em `OUTPUTS/...`.

Exemplos:

- [OUTPUTS/expts/01_ek100_avt.txt/0/checkpoint.pth](/media/storage/alana/AVT/OUTPUTS/expts/01_ek100_avt.txt/0/checkpoint.pth)
- [OUTPUTS/expts/14_ek100_vjepa2_tsn_action.txt/0/checkpoint.pth](/media/storage/alana/AVT/OUTPUTS/expts/14_ek100_vjepa2_tsn_action.txt/0/checkpoint.pth)

Os nomes relevantes são:

- `checkpoint.pth`: estado mais recente, usado para retomar treino.
- `checkpoint_best.pth`: melhor modelo validado, se a config habilitar isso.

O código que salva e recarrega checkpoint está em [func/train.py](/media/storage/alana/AVT/func/train.py:50).

### O que faz cada arquivo principal

#### Raiz do projeto

- [launch.py](/media/storage/alana/AVT/launch.py:1): launcher principal. Lê um arquivo em `expts/`, monta o comando Hydra e executa treino, teste, TensorBoard ou utilitários.
- [train_net.py](/media/storage/alana/AVT/train_net.py:1): ponto de entrada do treino. Inicializa seed, Hydra e chama a função em `func/`.
- [env.yaml](/media/storage/alana/AVT/env.yaml:1): ambiente Conda com dependências.
- [testh5.py](/media/storage/alana/AVT/testh5.py:1): script auxiliar simples para verificar arquivos HDF5.
- `submission.json`: artefato solto já gerado anteriormente; não é o caminho padrão de saída do pipeline.

#### `func/`

- [func/train.py](/media/storage/alana/AVT/func/train.py:1): loop principal de treino e avaliação, salvamento de checkpoints, retomada automática e geração de submissão.
- [func/train_eval_ops.py](/media/storage/alana/AVT/func/train_eval_ops.py:1): operações de treino/avaliação usadas pelo loop principal.

#### `datasets/`

- [datasets/data.py](/media/storage/alana/AVT/datasets/data.py:1): fábrica de datasets a partir das configs.
- [datasets/base_video_dataset.py](/media/storage/alana/AVT/datasets/base_video_dataset.py:1): base comum para datasets de vídeo.
- [datasets/epic_kitchens.py](/media/storage/alana/AVT/datasets/epic_kitchens.py:1): dataset EPIC-Kitchens.
- [datasets/breakfast_50salads.py](/media/storage/alana/AVT/datasets/breakfast_50salads.py:1): dataset 50Salads/Breakfast.
- [datasets/reader_fns.py](/media/storage/alana/AVT/datasets/reader_fns.py:1): leitores de vídeo/features.

#### `models/`

- [models/base_model.py](/media/storage/alana/AVT/models/base_model.py:1): monta o modelo final combinando backbone, agregador temporal, preditor de futuro e classificadores.
- [models/video_classification.py](/media/storage/alana/AVT/models/video_classification.py:1): backbones/classificação em vídeo.
- [models/temporal_aggregation.py](/media/storage/alana/AVT/models/temporal_aggregation.py:1): agregadores temporais.
- [models/future_prediction.py](/media/storage/alana/AVT/models/future_prediction.py:1): predição futura, incluindo AVT-head e variantes.
- [models/classifiers.py](/media/storage/alana/AVT/models/classifiers.py:1): cabeças de classificação.

#### `common/`

- [common/log.py](/media/storage/alana/AVT/common/log.py:1): logging, TensorBoard e W&B.
- [common/utils.py](/media/storage/alana/AVT/common/utils.py:1): utilitários distribuídos, save/load e helpers.
- [common/scheduler.py](/media/storage/alana/AVT/common/scheduler.py:1): warmup e schedulers.
- [common/transforms.py](/media/storage/alana/AVT/common/transforms.py:1): transforms de vídeo.
- [common/sampler.py](/media/storage/alana/AVT/common/sampler.py:1): samplers auxiliares.
- [common/cluster.py](/media/storage/alana/AVT/common/cluster.py:1): utilidades relacionadas a cluster.
- [common/wandb_writer.py](/media/storage/alana/AVT/common/wandb_writer.py:1): integração com W&B.

#### `loss_fn/`

- [loss_fn/mse.py](/media/storage/alana/AVT/loss_fn/mse.py:1): perda de regressão MSE.
- [loss_fn/multidim_xentropy.py](/media/storage/alana/AVT/loss_fn/multidim_xentropy.py:1): cross-entropy multidimensional.
- [loss_fn/simclr_infonce.py](/media/storage/alana/AVT/loss_fn/simclr_infonce.py:1): InfoNCE/SimCLR.

#### `conf/`

- [conf/config.yaml](/media/storage/alana/AVT/conf/config.yaml:1): configuração base do projeto.
- `conf/data/`: parâmetros de pré-processamento e loading.
- `conf/dataset/`: configurações dos datasets e caminhos.
- `conf/model/`: configs de backbone, agregador temporal, preditor e classificador.
- `conf/opt/`: otimização e scheduler.
- `conf/train_eval_op/`: operação de treino/avaliação e losses associadas.

#### `expts/`

Cada arquivo `.txt` em `expts/` é um conjunto de overrides Hydra para um experimento específico.

Exemplos:

- [expts/01_ek100_avt.txt](/media/storage/alana/AVT/expts/01_ek100_avt.txt:1): treino AVT para EK100.
- [expts/14_ek100_vjepa2_tsn_action.txt](/media/storage/alana/AVT/expts/14_ek100_vjepa2_tsn_action.txt:1): experimento mais recente com V-JEPA2 e geração de submissão habilitada.
- arquivos com sufixo `_test_testonly.txt` ou `_test_trainval.txt`: usados para inferência/submissão no teste.

#### `sample_scripts/`

- [sample_scripts/resize_epic_256px.sh](/media/storage/alana/AVT/sample_scripts/resize_epic_256px.sh:1): redimensiona vídeos do EPIC-Kitchens para altura 256px, seguindo o padrão esperado em várias configs.

### Como o launcher funciona

O fluxo padrão é:

1. você escolhe um arquivo em `expts/`;
2. `launch.py` lê esse arquivo;
3. ele monta os overrides Hydra;
4. chama `train_net.py`;
5. `train_net.py` executa a função principal em `func/train.py`.

Comandos principais:

```bash
python launch.py -c expts/01_ek100_avt.txt
python launch.py -c expts/01_ek100_avt.txt -l
python launch.py -c expts/01_ek100_avt.txt -g
python launch.py -c expts/01_ek100_avt.txt -t
```

Significado:

- `-c`: escolhe o experimento.
- `-l`: roda localmente usando as GPUs da máquina.
- `-g`: modo debug local, com menos paralelismo.
- `-t`: roda somente teste/avaliação.

### Como gerar o JSON de teste

No código atual, o JSON de teste é gerado pelo bloco de pós-treino em [func/train.py](/media/storage/alana/AVT/func/train.py:1100).

Para isso acontecer, a config precisa ter:

- `post_train.run_test_submission=true`
- um `dataset@dataset_test=...`

Quando isso está ativo, o pipeline faz:

1. carrega o checkpoint final ou o melhor checkpoint;
2. roda inferência no split de teste;
3. salva os logits em `results_test/`;
4. monta `challenge/test.json`;
5. empacota `challenge/submit.zip`.

Exemplo de config que já faz isso:

- [expts/14_ek100_vjepa2_tsn_action.txt](/media/storage/alana/AVT/expts/14_ek100_vjepa2_tsn_action.txt:1)

Saída esperada:

- `OUTPUTS/expts/<nome_do_experimento>/<run_id>/challenge/test.json`
- `OUTPUTS/expts/<nome_do_experimento>/<run_id>/challenge/submit.zip`

Exemplo real já presente no projeto:

- `OUTPUTS/expts/02_ek100_avt_tsn_test_testonly.txt/0/challenge/`

Se você quiser gerar o JSON com uma config existente, há dois caminhos práticos:

#### Usar uma config que já tenha submissão habilitada

```bash
python launch.py -c expts/14_ek100_vjepa2_tsn_action.txt -l
```

