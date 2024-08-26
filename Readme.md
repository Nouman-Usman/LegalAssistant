# ApnaWaqeel

ApnaWaqeel is a web application that integrates various AI and search functionalities to provide users with relevant information based on their queries. The application leverages web search, embeddings, and Pinecone for vector storage.

## Features

- Query processing and search
- Embedding generation
- Vector storage using Pinecone
- Reciprocal rank fusion for combining search results

## Requirements

- Python 3.7+
- Flask
- python-dotenv
- langchain-huggingface
- langchain-pinecone
- langchain-groq
- langchain-core
- requests
- pinecone-client

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/ApnaWaqeel.git
    cd ApnaWaqeel
    ```

2. Create a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    Create a `.env` file in the root directory and add your environment variables (e.g., API keys).

## Usage

1. Run the application:
    ```sh
    flask run
    ```

2. Access the application in your web browser at `http://127.0.0.1:5000`.

## Code Overview

### Main Functions

- **Query Processing**: Handles user queries, performs web search, generates embeddings, and stores results in Pinecone.
- **Reciprocal Rank Fusion**: Combines multiple search results to improve the relevance of the final output.

### Error Handling

The application includes basic error handling to return informative messages when exceptions occur.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries, please contact [yourname@example.com](mailto:yourname@example.com).