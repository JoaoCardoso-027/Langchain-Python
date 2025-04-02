# Langchain-Python

## Descrição

Este projeto implementa um sistema de Retrieval-Augmented Generation (RAG) utilizando o LangChain e o modelo LLaMA 2 via Ollama. O sistema permite realizar buscas contextuais em um documento PDF, extraindo informações relevantes e gerando respostas baseadas nos dados extraídos.

## Tecnologias Utilizadas

Python: Linguagem principal do projeto.

LangChain: Framework para composição de cadeias de modelos de IA.

Ollama: Para execução do modelo LLaMA 2 localmente.

FAISS: Base de dados vetorial para recuperação eficiente de documentos.

PyPDFLoader: Para carregamento e extração de texto de arquivos PDF.

## Explicação do Código

1. Carregamento e Processamento do PDF

A função get_documents_from_pdf() utiliza o PyPDFLoader para carregar o PDF e o RecursiveCharacterTextSplitter para dividir o texto em chunks menores.

2. Criação da Base Vetorial

A função create_db(docs) converte os documentos em embeddings usando o OllamaEmbeddings e armazena-os no FAISS para recuperação eficiente.

3. Construção da Cadeia de Recuperação e Geração de Respostas

A função create_chain(vectorStore) cria a cadeia de processamento, combinando:

Um modelo ChatOllama (LLaMA 2) para gerar respostas.

Um prompt estruturado para fornecer contexto.

Um retriever baseado na base vetorial.

4. Consulta e Resposta

A consulta é realizada através da chain.invoke() passando a pergunta desejada. O modelo responde com base no conteúdo recuperado.

Exemplo de Saída

Pergunta:

Tell me what you know about Gragas?

Saída (exemplo):

Gragas is a large and boisterous brewmaster from League of Legends known for his love of ale and his powerful drunken brawls...

## O projeto seguiu como base a seguinte estrutura:
![image](https://github.com/user-attachments/assets/b3ad496b-f57c-4e4a-8ddb-e0872f4ed330)

