import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import pandas as pd
import numpy as np
import torch
from services.retrieve import retrieve_relevant_resources, print_top_results_and_scores
from llm.get_response import ask

class GenRAG:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.load_embeddings()
        self.create_widgets()
        
    def setup_window(self):
        self.root.title("GenRAG - AI Document Assistant")
        self.root.geometry("900x700")
        self.root.configure(bg='#0d1117')
        self.root.resizable(True, True)
        
    def load_embeddings(self):
        try:
            embeddings_df_save_path = "data/text_chunks_and_embeddings_df.csv"
            df = pd.read_csv(embeddings_df_save_path)
            df["embedding"] = df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
            self.pages_and_chunks = df.to_dict(orient="records")
            self.embeddings = torch.tensor(np.array(df["embedding"].tolist()), dtype=torch.float32)
            self.embeddings_df_save_path = embeddings_df_save_path
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load embeddings: {str(e)}")
            
    def create_widgets(self):
        main_frame = tk.Frame(self.root, bg='#0d1117')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header
        header_frame = tk.Frame(main_frame, bg='#161b22', relief='flat', bd=1)
        header_frame.pack(fill='x', pady=(0, 20))
        
        title_frame = tk.Frame(header_frame, bg='#161b22')
        title_frame.pack(fill='x', padx=30, pady=20)
        
        title = tk.Label(title_frame, text="GenRAG", font=('Segoe UI', 28, 'bold'), 
                        fg='#58a6ff', bg='#161b22')
        title.pack(anchor='w')
        
        subtitle = tk.Label(title_frame, text="AI-Powered Document Intelligence", 
                           font=('Segoe UI', 14), fg='#8b949e', bg='#161b22')
        subtitle.pack(anchor='w', pady=(5, 0))
        
        author = tk.Label(title_frame, text="by Ashish Prasad", 
                         font=('Segoe UI', 10), fg='#6e7681', bg='#161b22')
        author.pack(anchor='w', pady=(2, 0))
        
        # Search
        search_frame = tk.Frame(main_frame, bg='#0d1117')
        search_frame.pack(fill='x', pady=(0, 20))
        
        search_label = tk.Label(search_frame, text="Ask your question:", 
                               font=('Segoe UI', 12, 'bold'), fg='#f0f6fc', bg='#0d1117')
        search_label.pack(anchor='w', pady=(0, 8))
        
        input_container = tk.Frame(search_frame, bg='#21262d', relief='flat', bd=1)
        input_container.pack(fill='x', ipady=2)
        
        self.query_entry = tk.Entry(input_container, font=('Segoe UI', 12), 
                                   bg='#21262d', fg='#f0f6fc', insertbackground='#58a6ff',
                                   relief='flat', bd=0)
        self.query_entry.pack(side='left', fill='x', expand=True, padx=15, pady=12)
        self.query_entry.bind('<Return>', lambda e: self.search_query())
        
        self.search_btn = tk.Button(input_container, text="🔍 Search", 
                                   font=('Segoe UI', 11, 'bold'), bg='#238636', 
                                   fg='white', relief='flat', bd=0, padx=25, pady=8,
                                   command=self.search_query, cursor='hand2')
        self.search_btn.pack(side='right', padx=15, pady=8)
        
        # Results
        results_label = tk.Label(main_frame, text="Results:", 
                                font=('Segoe UI', 12, 'bold'), fg='#f0f6fc', bg='#0d1117')
        results_label.pack(anchor='w', pady=(0, 8))
        
        results_container = tk.Frame(main_frame, bg='#161b22', relief='flat', bd=1)
        results_container.pack(fill='both', expand=True)
        
        self.results_text = scrolledtext.ScrolledText(
            results_container, font=('Consolas', 11), bg='#0d1117', fg='#e6edf3',
            insertbackground='#58a6ff', relief='flat', wrap='word',
            padx=20, pady=20, selectbackground='#264f78'
        )
        self.results_text.pack(fill='both', expand=True, padx=1, pady=1)
        
        # Status
        status_frame = tk.Frame(main_frame, bg='#161b22', height=30)
        status_frame.pack(fill='x', pady=(10, 0))
        status_frame.pack_propagate(False)
        
        self.status_var = tk.StringVar(value="Ready - Ask me anything about investing!")
        status_label = tk.Label(status_frame, textvariable=self.status_var,
                               font=('Segoe UI', 10), fg='#8b949e', bg='#161b22')
        status_label.pack(side='left', padx=15, pady=8)
        
        # Sample queries
        self.add_sample_queries(main_frame)
        
    def add_sample_queries(self, parent):
        samples_frame = tk.Frame(parent, bg='#0d1117')
        samples_frame.pack(fill='x', pady=(10, 0))
        
        samples_label = tk.Label(samples_frame, text="Try these questions:", 
                                font=('Segoe UI', 10), fg='#8b949e', bg='#0d1117')
        samples_label.pack(anchor='w')
        
        queries = ["What is value investing?", "How to analyze stocks?", 
                  "What is margin of safety?", "Investment principles"]
        
        buttons_frame = tk.Frame(samples_frame, bg='#0d1117')
        buttons_frame.pack(fill='x', pady=5)
        
        for query in queries:
            btn = tk.Button(buttons_frame, text=query, font=('Segoe UI', 9),
                           bg='#21262d', fg='#58a6ff', relief='flat', bd=0,
                           padx=12, pady=4, cursor='hand2',
                           command=lambda q=query: self.set_query(q))
            btn.pack(side='left', padx=(0, 8))
            
    def set_query(self, query):
        self.query_entry.delete(0, tk.END)
        self.query_entry.insert(0, query)
        
    def search_query(self):
        query = self.query_entry.get().strip()
        if not query:
            return
            
        self.search_btn.config(state='disabled', text='🔄 Searching...')
        self.status_var.set("Processing your question...")
        
        def search_thread():
            try:
                response = ask(query=query, embeddings=self.embeddings, 
                             pages_and_chunks=self.pages_and_chunks,
                             embeddings_df_save_path=self.embeddings_df_save_path)
                
                self.root.after(0, self.display_results, query, response)
                
            except Exception as e:
                self.root.after(0, self.show_error, str(e))
                
        threading.Thread(target=search_thread, daemon=True).start()
        
    def display_results(self, query, response):
        self.results_text.delete(1.0, tk.END)
        
        formatted_text = f"❓ Question: {query}\n\n"
        formatted_text += "🤖 AI Response:\n"
        formatted_text += "─" * 60 + "\n\n"
        formatted_text += response
        
        self.results_text.insert(1.0, formatted_text)
        
        self.search_btn.config(state='normal', text='🔍 Search')
        self.status_var.set("Search completed successfully!")
        
    def show_error(self, error_msg):
        messagebox.showerror("Error", f"Search failed: {error_msg}")
        self.search_btn.config(state='normal', text='🔍 Search')
        self.status_var.set("Error occurred - please try again")

if __name__ == "__main__":
    root = tk.Tk()
    app = GenRAG(root)
    
    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (450)
    y = (root.winfo_screenheight() // 2) - (350)
    root.geometry(f"900x700+{x}+{y}")
    
    root.mainloop()