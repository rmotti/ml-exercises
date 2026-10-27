# ml-exercises

Exercícios e projetos da disciplina de Machine Learning (2026.2).

Cada pasta é um projeto independente, criado a partir do template
[Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/),
com seu próprio ambiente Python (3.12, gerenciado com `uv`).

## Projetos

| Projeto | Aulas | Conteúdo |
|---|---|---|
| [`olist`](olist/) | 03 a 12 | Ciclo completo de um projeto de ML sobre o dataset público de e-commerce da Olist: qualidade dos dados, análise exploratória, engenharia de features, treino e avaliação, validação cruzada, inferência, explicabilidade com SHAP e active learning |
| [`deep_learning`](deep_learning/) | 13 e 14 | Introdução a redes neurais com TensorFlow/Keras: atividade de construção de uma rede neural, classificação de dados estruturados e classificação de imagens CIFAR-10 |
| [`classificador_imagens`](classificador_imagens/) | 16 a 18 | Pipeline de classificação de imagens CIFAR-10 com CNN: notebook, carga e normalização dos dados, treino, avaliação, gráficos e inferência |
| [`reconhecimento_facial`](reconhecimento_facial/) | 19 | Controle de acesso por reconhecimento facial na webcam com OpenCV e DeepFace |
| [`vaes_poc`](https://github.com/rmotti/vaes_poc) | 20 | Projeto em grupo, mantido em repositório próprio |

## Entregas por aula

Commit principal de cada entrega e, quando houver, os commits que a complementam.

| Aula | Entrega | Commit | Complementos |
|---|---|---|---|
| 03/04 | Estrutura inicial do projeto | [`ab48466`][ab48466] chore(olist): adiciona estrutura inicial do projeto | — |
| 04 | Notebook de qualidade dos dados | [`2ddabb9`][2ddabb9] feat(olist): adiciona notebook de qualidade dos dados e profiling de orders | [`7106990`][7106990], [`73660c6`][73660c6], [`e5fd2bb`][e5fd2bb] |
| 05 | Notebook com análise exploratória | [`91c28d3`][91c28d3] feat(olist): adiciona notebook com análise exploratória (EDA) | — |
| 06 | Engenharia de features | [`dd200db`][dd200db] feat(olist): features engineering | — |
| 07 | Modelo treinado | [`cf1b8a0`][cf1b8a0] feat(olist): add model training and evaluation | — |
| 08/09 | Validação do modelo com cross-validation | [`5aedf2a`][5aedf2a] refactor(olist): ajusta forma de validação do modelo com cross-validation | — |
| 10 | Inferência do modelo | [`210de39`][210de39] feat(olist): realiza inferência do modelo treinado | — |
| 11 | Explicabilidade com SHAP | [`aba712e`][aba712e] feat(olist): adiciona explicabilidade com SHAP | — |
| 12 | Exemplo de active learning | [`9dda385`][9dda385] feat(olist): implement active learning with Label Spreading example | — |
| 13 | Estrutura inicial do projeto de deep learning | [`bc60e7a`][bc60e7a] feat(deep_learning): initialize deep learning project config | — |
| 14 | Deep learning com Keras | [`468a421`][468a421] feat(deep_learning): implementa classificação de dados estruturados com keras | [`a95104a`][a95104a], [`7f9986c`][7f9986c] |
| 16 | Estrutura inicial do projeto de visão computacional | [`5cdf49f`][5cdf49f] chore(classificador_imagens): adiciona estrutura inicial do projeto de visão computacional | [`469c51c`][469c51c] |
| 17/18 | Notebook com o classificador de imagens | [`c794c82`][c794c82] feat(classificador_imagens): adiciona notebook do classificador de imagens CIFAR-10 | [`9ac23b3`][9ac23b3], [`6e8efea`][6e8efea], [`fbcb091`][fbcb091] |
| 19 | Solução do projeto de reconhecimento facial | [`c96e1a7`][c96e1a7] feat(reconhecimento_facial): add face recognition access control with deepface and webcam inference | [`9fec7fa`][9fec7fa] |
| 20 | Cópia do projeto realizado em grupo | [rmotti/vaes_poc](https://github.com/rmotti/vaes_poc) (repositório próprio do grupo) | — |

## Como rodar um projeto

```bash
cd <projeto>
make requirements          # instala as dependências com uv sync
source .venv/bin/activate
```

Os comandos específicos de cada projeto estão no `README.md` e no `Makefile` da pasta.

[ab48466]: https://github.com/rmotti/ml-exercises/commit/ab48466b4c60849b42d5b050c6063f279a9eb3d5
[2ddabb9]: https://github.com/rmotti/ml-exercises/commit/2ddabb9da3ea9072a8d19a9d7230b1b5dc5fe72a
[7106990]: https://github.com/rmotti/ml-exercises/commit/71069909291344b10731322fb5cb3188d245a465
[73660c6]: https://github.com/rmotti/ml-exercises/commit/73660c63a4d1038daa64491760a0fbe33218ea8f
[e5fd2bb]: https://github.com/rmotti/ml-exercises/commit/e5fd2bbda39ac6dcf22874d4651416bd6a66d565
[91c28d3]: https://github.com/rmotti/ml-exercises/commit/91c28d3f0e176d2561651a9faeb21c17cdb4320c
[dd200db]: https://github.com/rmotti/ml-exercises/commit/dd200db4363c6fc43bafc5b9d1780a672e7d8f5f
[cf1b8a0]: https://github.com/rmotti/ml-exercises/commit/cf1b8a0425a5ce29d82fe889e4d042850eb7ac88
[5aedf2a]: https://github.com/rmotti/ml-exercises/commit/5aedf2a5a1e741049166f438b8c5716cc4acf671
[210de39]: https://github.com/rmotti/ml-exercises/commit/210de390a1ef5097751cfc4a3ad6ae62a0bed2ca
[aba712e]: https://github.com/rmotti/ml-exercises/commit/aba712e8f0c7b706bf6b3447c00d10b09afcfd35
[9dda385]: https://github.com/rmotti/ml-exercises/commit/9dda385d8f5c083bed75f5e5f27de1e97c7899de
[bc60e7a]: https://github.com/rmotti/ml-exercises/commit/bc60e7a1ca8ef332363a9169036402899dad0e0a
[a95104a]: https://github.com/rmotti/ml-exercises/commit/a95104a58649057705806fcc01963a6a654fd2db
[468a421]: https://github.com/rmotti/ml-exercises/commit/468a421263ea4b6033b42a3be8adae612679bd49
[7f9986c]: https://github.com/rmotti/ml-exercises/commit/7f9986c7ea6b3583e93b593fbe55f83d14d66f1e
[5cdf49f]: https://github.com/rmotti/ml-exercises/commit/5cdf49fca2c515d3523233690f64014c39fb1334
[469c51c]: https://github.com/rmotti/ml-exercises/commit/469c51c19d3ee9aa8d7d196f93cde613718a757b
[c794c82]: https://github.com/rmotti/ml-exercises/commit/c794c8256e34c834eabeba68ac23f396ff169d5c
[9ac23b3]: https://github.com/rmotti/ml-exercises/commit/9ac23b3d231046c831681ef982ac0ef86ae101f6
[6e8efea]: https://github.com/rmotti/ml-exercises/commit/6e8efeae94027f6370db74e52ef1744b173ec00e
[fbcb091]: https://github.com/rmotti/ml-exercises/commit/fbcb09117b94a74feef07eda49b0ba16457a1c0e
[9fec7fa]: https://github.com/rmotti/ml-exercises/commit/9fec7fa187fc220e6d49b1f1d1406bc7dc09e0cc
[c96e1a7]: https://github.com/rmotti/ml-exercises/commit/c96e1a7d420af77c95dcbea9e48d919fac73b8ba
