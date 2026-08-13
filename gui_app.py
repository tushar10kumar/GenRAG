import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import pandas as pd
import numpy as np
import torch
from services.retrieve import retrieve_relevant_resources
from llm.get_response import ask

class GenRAGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GenRAG - AI Document Assistant")
        self.root.geometry("800x600")
        self.root.configure(bg='#1e1e1e')
        
        # Load embeddings on startup
        self.load_embeddings()
        self.setup_ui()
        
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
            
    def setup_ui(self):
        # Header
        header = tk.Frame(self.root, bg='#2d2d2d', height=80)
        header.pack(fill='x', padx=10, pady=10)
        
        title = tk.Label(header, text="GenRAG", font=('Arial', 24, 'bold'), 
                        fg='#4CAF50', bg='#2d2d2d')
        title.pack(side='left', padx=20, pady=20)
        
        subtitle = tk.Label(header, text="AI-Powered Document Assistant", 
                           font=('Arial', 12), fg='#888', bg='#2d2d2d')
        subtitle.pack(side='left', padx=(0, 20), pady=20)
        
        # Query input
        input_frame = tk.Frame(self.root, bg='#1e1e1e')
        input_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(input_frame, text="Ask a question:", font=('Arial', 12), 
                fg='white', bg='#1e1e1e').pack(anchor='w')
        
        query_frame = tk.Frame(input_frame, bg='#1e1e1e')
        query_frame.pack(fill='x', pady=5)
        
        self.query_entry = tk.Entry(query_frame, font=('Arial', 12), bg='#3d3d3d', 
                                   fg='white', insertbackground='white', relief='flat', bd=10)
        self.query_entry.pack(side='left', fill='x', expand=True, ipady=8)
        self.query_entry.bind('<Return>', lambda e: self.search_query())
        
        self.search_btn = tk.Button(query_frame, text="Search", font=('Arial', 12, 'bold'),
                                   bg='#4CAF50', fg='white', relief='flat', bd=0,
                                   padx=20, command=self.search_query)
        self.search_btn.pack(side='right', padx=(10, 0))
        
        # Results area
        results_frame = tk.Frame(self.root, bg='#1e1e1e')
        results_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        tk.Label(results_frame, text="Results:", font=('Arial', 12), 
                fg='white', bg='#1e1e1e').pack(anchor='w')
        
        self.results_text = scrolledtext.ScrolledText(results_frame, font=('Arial', 11),
                                                     bg='#2d2d2d', fg='white', 
                                                     insertbackground='white', relief='flat',
                                                     wrap='word', padx=15, pady=15)
        self.results_text.pack(fill='both', expand=True, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                             font=('Arial', 10), fg='#888', bg='#1e1e1e', anchor='w')
        status_bar.pack(fill='x', padx=20, pady=(0, 10))
        
    def search_query(self):
        query = self.query_entry.get().strip()
        if not query:
            return
            
        self.search_btn.config(state='disabled', text='Searching...')
        self.status_var.set("Processing query...")
        
        def search_thread():
            try:
                # Get response
                response = ask(query=query, embeddings=self.embeddings, 
                             pages_and_chunks=self.pages_and_chunks,
                             embeddings_df_save_path=self.embeddings_df_save_path)
                
                # Update UI in main thread
                self.root.after(0, self.display_results, query, response)
                
            except Exception as e:
                self.root.after(0, self.show_error, str(e))
                
        threading.Thread(target=search_thread, daemon=True).start()
        
    def display_results(self, query, response):
        self.results_text.delete(1.0, tk.END)
        
        # Format and display results
        formatted_text = f"Query: {query}\n\n"
        formatted_text += "=" * 50 + "\n\n"
        formatted_text += response
        
        self.results_text.insert(1.0, formatted_text)
        
        self.search_btn.config(state='normal', text='Search')
        self.status_var.set("Search completed")
        
    def show_error(self, error_msg):
        messagebox.showerror("Error", f"Search failed: {error_msg}")
        self.search_btn.config(state='normal', text='Search')
        self.status_var.set("Error occurred")

def main():
    root = tk.Tk()
    app = GenRAGApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()