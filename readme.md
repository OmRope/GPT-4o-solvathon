# **Stock Recommendation System**

Welcome to the Stock Recommendation System project! This repository features an advanced recommendation system that analyzes multiple data sources to provide actionable insights for stocks, advising users on whether to Buy, Sell, or Hold a stock. Additionally, we offer a Retrieval-Augmented Generation (RAG) pipeline chatbot that assists users with basic stock market queries, helping them deepen their market knowledge.

**📈 Project Overview**

This stock recommendation system leverages various aspects of stock analysis, including:

1. Historical Stock Data: Analyzes stock price trends, volatility, and technical indicators to assess the stock’s recent performance.
2. Recent News Sentiment Analysis: Scans recent news related to the stock, calculating sentiment scores to understand market sentiment and possible trends.
3. Financial Statements: Evaluates key financial metrics from recent statements (like revenue, profit margins, and EPS) to assess the company's financial health and growth potential.
Based on this multi-aspect analysis, the system recommends whether the stock should be bought, sold, or held.

**🔍 Stock Market Learning Assistant**

To aid users in learning about the stock market, we have implemented a chatbot based on a Retrieval-Augmented Generation (RAG) pipeline. This chatbot allows users to ask questions about various stock market concepts, strategies, terms, and more.

**Key Features of the Chatbot:**

RAG-based Responses: Combines a retrieval system with a generation model to provide relevant and accurate answers.
Stock Market Education: Enables users to gain a deeper understanding of terms like P/E ratios, market capitalization, dividend yields, and more.
Interactive and User-friendly: The chatbot provides a conversational and intuitive learning experience.

**🛠️ Features**

- Recommendation System:
- Analyzes stock trends, recent news, and financial statements.
- Offers Buy, Sell, or Hold recommendations.
- RAG Chatbot:
  - Query-based system that retrieves relevant stock market information.
  - Provides learning resources and explanations of stock market terms.

**⚙️ Setup Guidelines**
1. OpenAI API Key
- This project requires an OpenAI API key for generating chatbot responses. During the hackathon, participants were provided with an API key. Ensure that you use that key to proceed with this project. If you don’t have a key, you may need to acquire one by signing up on the OpenAI platform.<br>

  - Add the API key to your environment by creating a .env file in the project directory:
    ```OPENAI_API_KEY=<your_api_key_here>```

2. Install Dependencies
- Before running the project, install the required dependencies by executing:
  ```pip install -r requirements.txt```

3. Running the Application
- To start the recommendation system:
   ```python stock_recommendation.py```
- To use the chatbot:
   ```python chatbot.py```

6. Features in Action
- Recommendation System: Input stock ticker symbols to get recommendations.
- Chatbot: Ask questions like:
  - “What is a P/E ratio?”
  - “Should I buy Tesla stocks based on its current trend?”

**📝 Notes**
- Ensure you have access to the internet, as the OpenAI API requires external calls.
- The recommendation system uses financial APIs (e.g., Alpha Vantage, Yahoo Finance), so you may need API keys for those services as well.

**💡 Contributions**
- We welcome contributions to enhance this system! Feel free to submit issues or create a pull request for adding features or improving the chatbot's capabilities.
