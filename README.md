# 🎓 Inteligência Artificial no Contexto Educacional  
## Predição de Evasão Acadêmica

Este repositório apresenta um projeto de análise e implementação de modelos de **Machine Learning** voltados para a predição de evasão acadêmica, utilizando o dataset **OULAD (Open University Learning Analytics Dataset)**.

O trabalho foi desenvolvido como requisito da disciplina de **Inteligência Artificial (INF032G)** no curso de Tecnologia em Análise e Desenvolvimento de Sistemas do :contentReference[oaicite:0]{index=0} – Campus Salvador.

---

## 📌 Contextualização Teórica

A implementação foi fundamentada na análise comparativa dos seguintes estudos:

- **Evasão Universitária**  
  Identificação de padrões de desistência por meio de dados socioeconômicos e desempenho inicial  
  *(Souza et al., 2022)*

- **Desempenho Acadêmico**  
  Aplicação de arquiteturas Transformers para previsão de rendimento escolar  
  *(Rodrigues et al., 2024)*

- **Absenteísmo Docente**  
  Uso de Machine Learning para gestão de faltas na rede pública  
  *(Fernandes & Chiavegatto Filho, 2021)*

---

## 🛠️ Tecnologias e Modelos

O projeto foi desenvolvido em **Python 3.x**, utilizando bibliotecas amplamente adotadas na área de Ciência de Dados:

### 📚 Bibliotecas
- `pandas`
- `scikit-learn`

### 🤖 Modelos Implementados
- **K-Nearest Neighbors (KNN)**  
  Classificação baseada na proximidade entre instâncias

- **Random Forest (RF)**  
  Modelo de ensemble com alta robustez e redução de overfitting

- **Support Vector Machine (SVM)**  
  Busca do hiperplano ótimo para separação das classes

- **Voting Classifier (Ensemble)**  
  Combinação de modelos por votação majoritária (*hard voting*)

---

## 📊 Metodologia e Dataset

### 🗂️ Base de Dados
- **OULAD (Open University Learning Analytics Dataset)**
- Contém dados:
  - Demográficos
  - Socioeconômicos
  - Interações em ambientes virtuais de aprendizagem (VLE)

### 🎯 Variável Alvo
- `final_result`  
- Convertida para classificação binária:
  - **Evasão**
  - **Não evasão**

### ⚙️ Pré-processamento
- Normalização com `StandardScaler` (KNN e SVM)
- Codificação de variáveis categóricas com **One-Hot Encoding**

---

## 📈 Resultados Obtidos

- **Acurácia Geral:** entre **66% e 70%**

### 📌 Análise por Modelo
- **KNN:** melhor *recall* para a classe de evasão  
- **Random Forest:** desempenho mais equilibrado  
- **Ensemble:** maior estabilidade nas predições  

⚠️ **Desafio identificado:**  
O desbalanceamento da base impacta diretamente a capacidade de identificar alunos em risco.

---

## 🚀 Como Executar o Projeto

### 1. Preparar o dataset
Certifique-se de que o arquivo `data.csv` esteja no diretório:


dataset_files/


### 2. Instalar dependências
```bash
pip install pandas scikit-learn
```

### 3. Executar o script
```
python evasao.py
```


👨‍💻 Autor e Créditos

Estudante: Erick Rocha Nascimento
Docente: Marcelo Vera Cruz Diniz
Instituição: Instituto Federal de Educação, Ciência e Tecnologia da Bahia
