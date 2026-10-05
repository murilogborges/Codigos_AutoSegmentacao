# Processamento automático de pacientes DICOM

Este repositório automatiza uma cadeia de segmentação e consolidação de
RTSTRUCTs de um paciente. O ponto de entrada é `Process_DICOMFolder.sh`: ele
recebe uma pasta contendo uma série DICOM, identifica o grupo pelo nome da
pasta final e encaminha o paciente para o fluxo MAMA, META ou PELVE.

> **Importante:** TotalSegmentator e os scripts deste repositório são
> ferramentas de processamento, não substituem revisão clínica. Confirme os
> resultados no visualizador DICOM apropriado. Verifique também se os termos
> de uso dos modelos permitem o uso pretendido e se o ambiente cumpre as regras
> de segurança e privacidade dos dados de saúde.

## 1. Preparar o ambiente

### 1.1. TotalSegmentator

Consulte sempre as instruções de instalação e compatibilidade mais recentes
na [documentação oficial do TotalSegmentator](https://github.com/wasserth/TotalSegmentator).
O repositório oficial documenta instalação, versões do Python/PyTorch,
configuração de GPU/CPU, tarefas disponíveis, opções de linha de comando e
download dos modelos. A instalação padrão é feita via `pip`, mas o comando
exato pode depender do sistema operacional e da versão CUDA instalada; siga
as instruções oficiais em vez de copiar uma instalação de outra máquina.

Depois de ativar o ambiente virtual em que o TotalSegmentator foi instalado,
confirme que os comandos e módulos estão disponíveis:

```bash
command -v TotalSegmentator
TotalSegmentator --help
python3 -c 'import cv2, numpy, pydicom, rt_utils, scipy, shapely'
```

O lançador usa `python3` para chamar os utilitários Python. Depois de instalar
o TotalSegmentator conforme a documentação oficial, instale os pacotes
adicionais usados pelos scripts no mesmo ambiente que fornece o `python3`
encontrado no `PATH`:

```bash
python3 -m pip install numpy pydicom scipy shapely rt-utils opencv-python
```

O TotalSegmentator também depende de PyTorch e de arquivos de modelo; use as
instruções oficiais para configurá-los corretamente no computador, em
particular quando houver GPU/CUDA. O primeiro uso pode baixar pesos e exigir
acesso à rede. Não execute uma instalação nova diretamente em um ambiente
clínico sem antes validar a versão e as permissões aplicáveis.

### 1.2. Arquivos de entrada

Prepare uma pasta por paciente e aponte para a pasta que contém a série de
imagens DICOM a processar. Use uma aquisição CT compatível com as tarefas
configuradas; não misture pacientes nem séries ou aquisições diferentes na
mesma entrada. Confira previamente se a série está completa e se corresponde
ao estudo pretendido.

O nome da **pasta final informada** precisa conter exatamente um dos tokens
`MAMA`, `META` ou `PELVE`, sem diferenciar maiúsculas de minúsculas. Por
exemplo, `MAMA_Paciente01`, `META01` e `pelve_01` são reconhecidos. Não inclua
mais de um identificador de grupo no mesmo nome: por exemplo, `META_MAMA01` é
ambíguo e será recusado. A classificação usa o nome da pasta final, não o
nome de uma pasta ancestral.

Para MAMA, um indicador de lateralidade adjacente ao identificador numérico
também pode estar antes ou depois do número do paciente. O lançador reconhece
`D`, `DIR` ou `LD` como direito e `E`, `ESQ` ou `LE` como esquerdo,
ignorando `_`, hífenes, espaços e maiúsculas/minúsculas. Exemplos:

| Nome da pasta | Saída final |
|---|---|
| `MAMAD01` ou `MAMADIR01` | `GrupoMAMA_LD.dcm` |
| `MAMA01D`, `MAMA01LD` ou `MAMA01_LD` | `GrupoMAMA_LD.dcm` |
| `MAMAE01` ou `MAMAESQ01` | `GrupoMAMA_LE.dcm` |
| `MAMA01E`, `MAMA01LE` ou `MAMA01_LE` | `GrupoMAMA_LE.dcm` |
| `MAMA01` | `GrupoMAMA.dcm` |

Se a lateralidade aparecer dos dois lados do número e os indicadores forem
conflitantes, o lançador aborta por ambiguidade, sem iniciar segmentações.
Caso nenhum indicador seja encontrado, usa o arquivo MAMA sem sufixo.

As saídas RTSTRUCT individuais de cada tarefa podem permanecer na pasta entre
execuções: se um arquivo esperado, como `TOTAL_META.dcm`, já existir e não
estiver vazio, a tarefa correspondente será pulada e o arquivo reutilizado.
Um arquivo final consolidado já existente, um RTSTRUCT de tarefa vazio, um
resíduo NIfTI sem o RTSTRUCT correspondente ou um arquivo intermediário
derivado preexistente causam interrupção para evitar sobrescrita acidental.
Para refazer uma tarefa, mova ou remova manualmente apenas os arquivos de
saída correspondentes a ela e os artefatos derivados afetados.

## 2. Executar o lançador

Na pasta deste repositório, execute:

```bash
./Process_DICOMFolder.sh "/home/borges/META01/"
./Process_DICOMFolder.sh "/home/borges/MAMA01/"
./Process_DICOMFolder.sh "/home/borges/PELVE01/"
```

Aspas em torno do caminho são recomendadas, especialmente se ele tiver espaços.
Também é possível fornecer um caminho relativo. Os scripts Python auxiliares
são encontrados relativamente ao próprio `Process_DICOMFolder.sh`; o comando
pode ser iniciado de qualquer diretório, desde que o Bash, `python3`,
`TotalSegmentator` e os pacotes Python estejam disponíveis no `PATH`/ambiente
ativo.

O fluxo é sequencial. Se uma tarefa de segmentação falhar ou não gerar o
RTSTRUCT esperado, a tarefa é omitida, seus arquivos parciais com o mesmo
nome-base (`.dcm`, `.nii` e `.nii.gz`) são removidos, e o processamento
continua nas demais tarefas. Falhas nas etapas derivadas ou na consolidação
são informadas; examine as mensagens no terminal antes de tentar novamente.

## 3. Etapas comuns de cada execução

1. **Validar entrada e grupo.** O lançador confirma que a pasta existe e
   identifica exatamente um dos três grupos aceitos.
2. **Verificar ferramentas.** Confirma a presença do comando `TotalSegmentator`,
   de `python3` e dos pacotes Python usados pelos utilitários.
3. **Proteger e reutilizar resultados.** Para cada tarefa, se o RTSTRUCT
   correspondente já existir e não estiver vazio, ele é reutilizado e a
   chamada ao TotalSegmentator é ignorada. Saídas ausentes são processadas.
   Um RTSTRUCT vazio ou resíduos NIfTI sem o RTSTRUCT correspondente são
   informados como conflito, sem sobrescrever arquivos.
4. **Segmentar.** Os scripts chamam TotalSegmentator diretamente sobre a pasta
   DICOM. A saída normal do TotalSegmentator fica oculta para reduzir o ruído
   no terminal; se uma tarefa falhar ou não produzir o DICOM esperado, são
   exibidas as últimas mensagens relevantes do comando.
5. **Derivar estruturas adicionais.** Conforme o grupo, os utilitários deste
   repositório separam estruturas, constroem máscaras derivadas ou unificam
   ROIs.
6. **Consolidar e verificar.** `unir_rtstructs.py` recebe apenas os RTSTRUCTs
   efetivamente gerados e preparados. Todas as entradas selecionadas ainda
   precisam ter referências compatíveis de paciente/estudo, série e Frame of
   Reference. A consolidação pode ser parcial se tarefas foram omitidas; se
   nenhuma entrada aproveitável restar, o fluxo termina com erro explícito.
   O script também verifica a consistência das sequências de ROIs, contornos
   e observações DICOM no arquivo final.

Durante o processamento, os scripts exibem o percentual concluído do fluxo
completo (segmentações, etapas derivadas e consolidação). Esse percentual
avança ao final de cada etapa; por isso, uma tarefa de segmentação longa pode
permanecer algum tempo no mesmo valor.

## 4. Fluxo MAMA, em detalhe

O lançador chama `MAMA_Segmentations.sh <pasta_DICOM>`.

### 4.1. Segmentações TotalSegmentator

As chamadas são executadas em sequência. `-i` recebe a pasta DICOM, `-o`
define o nome-base de saída, `-ta` escolhe a tarefa, e
`-ot dicom_rtstruct` solicita saída RTSTRUCT DICOM. As opções adicionais
`-ho` e `-bs` permanecem conforme configuradas no pipeline; consulte
`TotalSegmentator --help` para o significado associado à versão instalada.

| Tarefa (`-ta`) | Nome-base de saída | O que o pipeline usa |
|---|---|---|
| `total` com `--roi_subset` | `TOTAL_MAMA.dcm` | Subconjunto de órgãos, pulmões lobares, esôfago, traqueia, tireoide, duodeno, vértebras, úmero, medula, costelas, esterno e cartilagens costais configurado no script. |
| `breasts` | `BREASTS.dcm` | ROI `breast`, usada para separação por lado. |
| `heartchambers_highres` | `HEART.dcm` | Câmaras cardíacas, miocárdio, aorta e artéria pulmonar; entrada de `NothingBreaksLikeAHeart.py`. |
| `coronary_arteries` | `CORO.dcm` | Artérias coronárias. |
| `body` | `BODY.dcm` | Segmentação de entrada obrigatória para `IHaveABody.py`, que produz `CORPO.dcm`; a ROI `corpo` é obrigatória no resultado consolidado. |
| `tissue_4_types` | `TISSUE.dcm` | A tarefa é executada e o arquivo fica disponível como intermediário, mas nunca é incluído na consolidação. |
| `headneck_muscles` | `MUSC.dcm` | Músculos de cabeça/pescoço. |
| `liver_lesions` | `LIVER.dcm` | Lesões hepáticas. |

Cada tarefa é independente: se, por exemplo, `HEAD.dcm` já estiver presente e
não vazio, `head_glands_cavities` é reutilizada e não é executada novamente.
Caso contrário, se a tarefa `head_glands_cavities` retornar
“Crop is empty” e não produzir `HEAD.dcm`, o script remove os resíduos
`HEAD.nii`/`HEAD.nii.gz` (e eventual `HEAD.dcm` parcial) silenciosamente e
segue para as próximas tarefas. As etapas derivadas `BreastSplit.py` e
`NothingBreaksLikeAHeart.py` só são executadas quando a
respectiva entrada DICOM foi gerada. `IHaveABody.py` é obrigatória: se
`BODY.dcm` não for gerado ou não puder produzir `CORPO.dcm`, o fluxo termina
sem criar um consolidado que omita a estrutura corpo.

Se um `CORPO.dcm` não vazio já existir de uma execução anterior, ele é
reutilizado no consolidado e tanto a tarefa `body` quanto `IHaveABody.py` são
puladas. Isso permite retomar a consolidação sem recalcular ou sobrescrever
essa estrutura.

### 4.2. Estruturas derivadas e consolidação

1. Se `BREASTS.dcm` existir, `BreastSplit.py` sempre o processa novamente e
   recria `BREASTS_separado.dcm`, mesmo que a saída derivada já exista. Se
   `BREASTS.dcm` não existir, a tarefa TotalSegmentator `breasts` é executada
   antes. O utilitário lê a máscara `breast`, separa os lados e cria um
   RTSTRUCT que contém `Breast_right` e `Breast_left`.
2. `NothingBreaksLikeAHeart.py HEART.dcm <pasta_DICOM>` combina as máscaras
   cardíacas e cria ou sobrescreve `Cardiac_area.dcm`. A gravação é atômica:
   se o reprocessamento falhar antes de terminar, o arquivo anterior é
   preservado.
3. `IHaveABody.py BODY.dcm <pasta_DICOM>` combina as estruturas do RTSTRUCT
   `BODY.dcm` em uma máscara chamada `corpo`, salva em `CORPO.dcm`.
4. `unir_rtstructs.py` recebe `Cardiac_area.dcm`, `BREASTS_separado.dcm`,
   `TOTAL_MAMA.dcm`, `CORO.dcm`, `CORPO.dcm`, `MUSC.dcm` e `LIVER.dcm`, e
   grava `GrupoMAMA.dcm`, `GrupoMAMA_LD.dcm` ou `GrupoMAMA_LE.dcm`, de acordo
   com o indicador de lateralidade identificado no nome da pasta.

`HEART.dcm` e `BODY.dcm` alimentam as transformações anteriores; não são
passados diretamente ao unificador. `TISSUE.dcm` também é gerado, mas não está
na lista de entradas atuais para o arquivo consolidado MAMA.

Se um RTSTRUCT de entrada não contiver `StructureSetROISequence`, o unificador
ignora esse arquivo e continua com os demais RTSTRUCTs válidos.

## 5. Fluxo META, em detalhe

O lançador chama `META_Segmentations.sh <pasta_DICOM> [arquivo_final]`.

### 5.1. Segmentações TotalSegmentator

| Tarefa (`-ta`) | Nome-base de saída | O que o pipeline usa |
|---|---|---|
| `total` | `TOTAL_META.dcm` | Segmentações gerais; também é consultado para decidir se há ROI `brain`. |
| `body` | `BODY.dcm` | Entrada obrigatória de `IHaveABody.py`; a máscara resultante `CORPO.dcm` integra o arquivo final. |
| `tissue_4_types` | `TISSUE.dcm` | Tarefa executada; o resultado fica como intermediário e não é passado ao unificador. |
| `head_glands_cavities` | `HEAD.dcm` | Glândulas e cavidades da cabeça/pescoço. |
| `liver_lesions` | `LIVER.dcm` | Lesões hepáticas. |

Para cada tarefa, se o respectivo `<nome-base>.dcm` já existir e não estiver
vazio, ele é reutilizado e a chamada é pulada. Se não existir, o pipeline usa
`-i <pasta> -o <pasta>/<nome-base> -ta <tarefa> -ot dicom_rtstruct -ho -bs`.
A saída só é incluída na consolidação se a tarefa terminar sem erro e gerar
um `.dcm` não vazio. Caso contrário, os arquivos parciais `.dcm`, `.nii` e
`.nii.gz` daquele nome-base são removidos sem mensagens e as tarefas
restantes continuam.

### 5.2. Decisão condicional para tarefas cerebrais

Depois de gerar `TOTAL_META.dcm`, o script abre o RTSTRUCT com `pydicom` e
procura o nome exato de ROI `brain` em `StructureSetROISequence`:

- Se encontrar `brain`, tenta executar `brain_structures` → `BRAIN.dcm`,
  `head_muscles` → `TONGUE.dcm` e `oculomotor_muscles` → `OPTICS.dcm`.
  Cada saída é incluída individualmente se gerada; a falha de uma não impede
  as demais.
- Se não encontrar `brain`, ou `TOTAL_META.dcm` não tiver sido gerado, informa
  que as tarefas cerebrais foram omitidas.
- Se não conseguir ler/verificar o DICOM, informa o problema e omite somente
  as tarefas cerebrais.

### 5.3. Corpo e arquivo final

`IHaveABody.py BODY.dcm <pasta_DICOM>` combina as ROIs rasterizáveis do
RTSTRUCT `BODY.dcm` e gera `CORPO.dcm`. Contornos/ROIs individuais inválidos
são ignorados silenciosamente para permitir a criação da máscara com as
estruturas válidas restantes; se nenhuma máscara aproveitável restar, a etapa falha.
Essa etapa é obrigatória: sem `CORPO.dcm`, não é produzido um consolidado
incompleto. Um `CORPO.dcm` válido de execução anterior é reutilizado. O
unificador recebe `CORPO.dcm` junto com `TOTAL_META.dcm`, `HEAD.dcm`, as
entradas cerebrais (se criadas) e `LIVER.dcm`, e exige que a ROI `corpo`
esteja presente no RTSTRUCT final. `BODY.dcm` é somente entrada do utilitário
e não é passado diretamente ao unificador; `TISSUE.dcm` é gerado, mas também
não é passado à união. Por padrão, o resultado é `GrupoMETA.dcm`.

## 6. Fluxo PELVE

PELVE usa as mesmas segmentações, a mesma verificação opcional de `brain` e a
mesma geração de `CORPO.dcm` do fluxo META. A diferença é o nome do resultado:
`Process_DICOMFolder.sh` chama
`META_Segmentations.sh <pasta_DICOM> <pasta_DICOM>/GrupoPELVE.dcm`, portanto
o consolidado final é `GrupoPELVE.dcm`.

Também é possível chamar diretamente o script compartilhado e informar outro
caminho final que ainda não exista:

```bash
bash META_Segmentations.sh "/dados/PELVE01" "/dados/PELVE01/GrupoPELVE.dcm"
```

## 7. Arquivos gerados e comportamento em caso de erro

Todos os resultados ficam na pasta informada. Os nomes-base de TotalSegmentator
produzem arquivos com extensão `.dcm`. Os principais arquivos finais são:

| Grupo | Arquivo consolidado |
|---|---|
| MAMA sem lateralidade indicada | `GrupoMAMA.dcm` |
| MAMA direita | `GrupoMAMA_LD.dcm` |
| MAMA esquerda | `GrupoMAMA_LE.dcm` |
| META | `GrupoMETA.dcm` |
| PELVE | `GrupoPELVE.dcm` |

Arquivos intermediários incluem os RTSTRUCTs de cada tarefa, `CORPO.dcm` e,
para MAMA, `BREASTS_separado.dcm` e `Cardiac_area.dcm`. Eles não são apagados
ao final; podem ser úteis para inspeção e rastreabilidade.

Quando o arquivo final é produzido, a única saída do pipeline é o indicador
de progresso. Mensagens normais, diagnósticos de tarefas opcionais e avisos
de contornos/estruturas ignorados ficam ocultos. O processamento cardíaco
ignora contornos que falhem na validação ou na rasterização OpenCV e continua
com os contornos restantes. Internamente, RTSTRUCTs das
tarefas existentes e não vazios são reutilizados sem nova inferência; se uma
tarefa falhar ou não gerar `.dcm`, seus arquivos parciais `.dcm`, `.nii` e
`.nii.gz` são removidos silenciosamente e as tarefas seguintes continuam. Se ainda for
possível consolidar o resultado, a falha daquela tarefa não é impressa. Se
não for possível gerar o arquivo final, o script informa o erro e mostra o
diagnóstico relevante. Não remove saídas de outras tarefas nem NIfTIs com
outros nomes. Uma saída final já existente continua sendo motivo para
interromper.

O unificador exige que os RTSTRUCTs de entrada apontem para a mesma série e
Frame of Reference e que sejam compatíveis quanto a paciente/estudo. Isso é
intencional: não combine saídas de pacientes, aquisições ou referências
espaciais distintas. Arquivos de entrada sem `StructureSetROISequence` são
ignorados; se nenhum RTSTRUCT aproveitável restar, não será possível gerar o
arquivo final. Contornos individuais inválidos são ignorados sem
mensagens; as demais ROIs continuam sendo processadas. Avisos de estruturas
derivadas opcionais, como a impossibilidade de criar `tronco`, também ficam
ocultos no modo normal. Erros fatais só são apresentados quando impedem a
geração do arquivo consolidado.

No RTSTRUCT consolidado, a sequência começa por `corpo`, `mama_D`, `mama_E`,
`mama_D_aval` e `mama_E_aval` (quando disponíveis), seguida por `pulmao_D`,
`pulmao_E`, `pulmoes` e `area_cardiaca` (quando disponíveis). `area_cardiaca`
usa uma cor vermelho-rosa mais viva para facilitar sua identificação. As
demais estruturas originais e derivadas são ordenadas da posição mais cranial
à mais caudal, usando a coordenada superior-inferior dos contornos DICOM. Ao final ficam
agrupadas as estruturas da coluna vertebral, ordenadas por nível: cervicais
(C1 em direção a C7), torácicas (T1 em direção a T12), lombares (L1 em direção
aos níveis disponíveis) e sacrais. Estruturas vertebrais sem nível identificável
ficam depois dos níveis nomeados, mas ainda no bloco final da coluna.
Quando o unificador identifica uma ROI direita e sua correspondente esquerda,
as duas são mantidas lado a lado na sequência; os pares são ordenados pela
posição cranial-caudal média das duas estruturas, com a direita antes da
esquerda. Avisos por contornos individuais inválidos são suprimidos enquanto
os contornos válidos continuam sendo processados. Avisos de estruturas
derivadas opcionais, como a impossibilidade de criar `tronco`, não aparecem
no processamento automático. Erros fatais de leitura, referências DICOM
incompatíveis ou ausência de contornos aproveitáveis são mostrados quando
impedem gerar o arquivo final.

## 8. Opções, licenças, privacidade e limitações

- Cada `-ta` seleciona uma tarefa/modelo diferente do TotalSegmentator.
  `--roi_subset` limita quais classes são segmentadas pela tarefa `total`;
  confira a lista mantida em `MAMA_Segmentations.sh` antes de alterá-la.
- A página oficial informa que as tarefas `heartchambers_highres` e
  `tissue_4_types` requerem uma licença para uso comercial; há licenças
  acadêmicas não comerciais disponíveis segundo os termos do projeto. Confirme
  a licença vigente e os direitos de uso para **cada tarefa** no
  [repositório oficial](https://github.com/wasserth/TotalSegmentator) e na
  [página de licenças acadêmicas](https://backend.totalsegmentator.com/license-academic/).
- O TotalSegmentator pode enviar estatísticas de uso anônimas. A documentação
  oficial descreve como desativá-las no arquivo de configuração
  `~/.totalsegmentator/config.json`; consulte-a se a política de dados do
  ambiente exigir isso.
- Não envie imagens, identificadores ou outros dados de saúde a serviços
  externos sem autorização e aprovação institucional. O download dos pesos e
  o uso de serviços online têm considerações de privacidade próprias.
- Resultados automáticos podem conter erros, inclusive por diferenças de
  protocolo, cobertura anatômica, artefatos, contraste ou qualidade da imagem.
  Não interprete uma segmentação como diagnóstico nem como contorno validado
  sem revisão adequada.
- `unir_rtstructs.py` traduz nomes, filtra algumas ROIs, cria estruturas
  derivadas (por exemplo, pulmões combinados, margem da medula e avaliações
  anatômicas) e escreve metadados para compatibilidade com o fluxo atual. O
  resultado consolidado não é uma cópia integral de todos os intermediários.

## 9. Links oficiais e referências bibliográficas

### Links

- [TotalSegmentator — repositório, instalação, tarefas, opções e referências](https://github.com/wasserth/TotalSegmentator)
- [Pesquisa de tarefas/estruturas do TotalSegmentator](https://backend.totalsegmentator.com/find-task/)
- [Extensão TotalSegmentator para 3D Slicer](https://github.com/lassoan/SlicerTotalSegmentator)
- [nnU-Net — implementação e documentação](https://github.com/MIC-DKFZ/nnUNet)

### Referências bibliográficas (formato ABNT)

As referências abaixo são apresentadas em formato bibliográfico ABNT, com
acesso direto pelo DOI. O TotalSegmentator recomenda citar também o nnU-Net,
no qual o projeto se baseia. Os artigos específicos por tarefa são incluídos
quando a documentação oficial do TotalSegmentator os associa ao módulo
utilizado no pipeline.

1. WASSERTHAL, Jakob et al. TotalSegmentator: robust segmentation of 104
   anatomic structures in CT images. *Radiology: Artificial Intelligence*,
   v. 5, n. 5, e230024, 2023. DOI:
   [10.1148/ryai.230024](https://doi.org/10.1148/ryai.230024).
   Referência geral do TotalSegmentator e da tarefa `total`.
2. ISENSEE, Fabian et al. nnU-Net: a self-configuring method for deep
   learning-based biomedical image segmentation. *Nature Methods*, v. 18,
   n. 2, p. 203-211, 2021. DOI:
   [10.1038/s41592-020-01008-z](https://doi.org/10.1038/s41592-020-01008-z).
   Método-base recomendado pelo próprio TotalSegmentator.
3. WALTER, Alexandra et al. Segmentation of 71 anatomical structures
   necessary for the evaluation of guideline-conforming clinical target
   volumes in head and neck cancers. *Cancers*, v. 16, n. 2, art. 415, 2024.
   DOI: [10.3390/cancers16020415](https://doi.org/10.3390/cancers16020415).
   Referência indicada pelo TotalSegmentator para `head_glands_cavities` e
   `headneck_muscles`.
4. NICOLI, Andrew Phillip et al. Liver segment and lesion segmentation on CT
   and MRI: an open-source contribution to TotalSegmentator. *Journal of
   Digital Imaging*, v. 39, n. 4, p. 3508-3523, 2026. Publicado online em
   24 out. 2025. DOI:
   [10.1007/s10278-025-01716-y](https://doi.org/10.1007/s10278-025-01716-y).
   Referência indicada para `liver_lesions`.
5. CAI, Jason C. et al. Fully automated segmentation of head CT neuroanatomy
   using deep learning. *Radiology: Artificial Intelligence*, v. 2, n. 5,
   e190183, 2020. DOI:
   [10.1148/ryai.2020190183](https://doi.org/10.1148/ryai.2020190183).
   Referência indicada pelo TotalSegmentator como base parcial de
   `brain_structures`.

Para `body`, `breasts`, `heartchambers_highres`, `tissue_4_types`,
`coronary_arteries`, `head_muscles` e `oculomotor_muscles`, a lista oficial de
tarefas consultada não indica um artigo adicional específico junto a cada
módulo. Cite a referência geral do TotalSegmentator e não associe a esses
módulos artigos que a documentação não aponta. Confirme as instruções de
citação na versão instalada antes de publicar, pois a lista de tarefas e
referências pode mudar.