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

## Como rodar um projeto

```bash
cd <projeto>
make requirements          # instala as dependências com uv sync
source .venv/bin/activate
```

Os comandos específicos de cada projeto estão no `README.md` e no `Makefile` da pasta.
