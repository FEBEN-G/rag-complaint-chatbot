"""
Interactive Complaint Analysis Chatbot - Task 4
Gradio-based UI for the RAG-powered complaint analysis system.
"""
import gradio as gr
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.rag_pipeline import RAGPipeline


# Initialize RAG pipeline
print("Initializing RAG Pipeline...")
rag = RAGPipeline(
    vector_store_path='vector_store',
    collection_name='complaint_embeddings_full',
    embedding_model='sentence-transformers/all-MiniLM-L6-v2'
)
print("✅ RAG Pipeline ready!")


def format_sources(sources):
    """Format sources for display."""
    if not sources:
        return "No sources found."
    
    formatted = []
    for i, source in enumerate(sources, 1):
        metadata = source['metadata']
        relevance = (1 - source['distance']) * 100
        
        formatted.append(
            f"**Source {i}** (Relevance: {relevance:.1f}%)\n"
            f"- **Product:** {metadata.get('product', 'Unknown')}\n"
            f"- **Issue:** {metadata.get('issue', 'Unknown')}\n"
            f"- **Company:** {metadata.get('company', 'Unknown')}\n"
            f"- **Date:** {metadata.get('date_received', 'Unknown')}\n\n"
            f"*Complaint Excerpt:*\n"
            f"> {source['text'][:300]}...\n"
        )
    
    return "\n\n---\n\n".join(formatted)


def chat(message, history, n_results, product_filter):
    """
    Process user message and return response.
    
    Args:
        message: User's question
        history: Chat history
        n_results: Number of sources to retrieve
        product_filter: Product category filter
        
    Returns:
        Updated history and sources
    """
    if not message.strip():
        return history, "Please enter a question."
    
    # Apply product filter
    filter_value = None if product_filter == "All Products" else product_filter.lower()
    
    # Generate answer
    response = rag.generate_answer(
        query=message,
        n_results=n_results,
        product_filter=filter_value,
        use_llm=False
    )
    
    # Format answer
    answer = response['answer']
    
    # Update history
    history.append((message, answer))
    
    # Format sources
    sources_text = format_sources(response['sources'])
    
    return history, sources_text


def clear_chat():
    """Clear chat history."""
    return [], ""


# Create Gradio interface
with gr.Blocks(
    title="CrediTrust Complaint Analysis Chatbot",
    theme=gr.themes.Soft()
) as app:
    
    gr.Markdown(
        """
        # 🏦 CrediTrust Complaint Analysis Chatbot
        
        Ask questions about customer complaints across our financial products.
        The AI will retrieve relevant complaints and provide synthesized insights.
        
        **Example Questions:**
        - Why are people unhappy with Credit Cards?
        - What are the most common complaints about personal loans?
        - What problems do customers report with savings accounts?
        - What issues arise with money transfers?
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            # Chat interface
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500,
                show_label=True,
                avatar_images=(None, "🤖")
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask about customer complaints...",
                    scale=4,
                    lines=2
                )
                submit_btn = gr.Button("Ask", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("Clear Conversation", variant="secondary")
        
        with gr.Column(scale=1):
            # Settings
            gr.Markdown("### ⚙️ Settings")
            
            n_results = gr.Slider(
                minimum=3,
                maximum=10,
                value=5,
                step=1,
                label="Number of Sources",
                info="How many complaint excerpts to retrieve"
            )
            
            product_filter = gr.Dropdown(
                choices=[
                    "All Products",
                    "Credit Card",
                    "Personal Loan",
                    "Savings Account",
                    "Money Transfer"
                ],
                value="All Products",
                label="Product Filter",
                info="Filter by specific product category"
            )
            
            gr.Markdown("### 📚 Retrieved Sources")
            sources_display = gr.Markdown(
                value="Sources will appear here after asking a question.",
                label="Sources"
            )
    
    gr.Markdown(
        """
        ---
        ### 📊 About This System
        
        This chatbot uses **Retrieval-Augmented Generation (RAG)** to analyze customer complaints:
        
        1. **Semantic Search**: Your question is converted to an embedding and matched against 1.3M+ complaint chunks
        2. **Retrieval**: The most relevant complaints are retrieved based on semantic similarity
        3. **Synthesis**: The system synthesizes insights from the retrieved complaints
        4. **Source Citation**: All answers are backed by actual customer complaints
        
        **Technology Stack:**
        - Embeddings: `all-MiniLM-L6-v2` (384-dim)
        - Vector Store: ChromaDB
        - Framework: LangChain
        - UI: Gradio
        
        **Data Source:** Consumer Financial Protection Bureau (CFPB) Complaint Database
        """
    )
    
    # Event handlers
    submit_btn.click(
        fn=chat,
        inputs=[msg, chatbot, n_results, product_filter],
        outputs=[chatbot, sources_display]
    ).then(
        lambda: "",
        outputs=[msg]
    )
    
    msg.submit(
        fn=chat,
        inputs=[msg, chatbot, n_results, product_filter],
        outputs=[chatbot, sources_display]
    ).then(
        lambda: "",
        outputs=[msg]
    )
    
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, sources_display]
    )


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
