import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from server.rl.state_encoder import encode_state  # . Import the actual encoder

# --- PPO Policy Architecture (MATCHED TO TRAINER) ---
class PPOPolicy(nn.Module):
    def __init__(self, state_dim=388, action_dim=3): 
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )

        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        # Ensure input is a tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # If it's a 1D tensor (single state), add batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = self.shared(x)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value

    def get_action(self, state_dict):
        """
        Takes the raw dictionary from the UI, encodes it to 388-dim,
        and returns the best action index.
        """
        # 1. Convert dict to the 388-dim vector using your trainer's logic
        state_vec = encode_state(state_dict)
        state_tensor = torch.tensor(state_vec, dtype=torch.float32)

        # 2. Inference
        with torch.no_grad():
            logits, _ = self.forward(state_tensor)
            probs = F.softmax(logits, dim=-1)
            action = torch.argmax(probs, dim=-1).item()
            
        return action, None, None

# --- Vector Memory Manager (Kept for UI compatibility) ---
class MemoryManager:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.memory_texts = []
        self.memory_vectors = None

    def add(self, text):
        new_vec = self.model.encode([text], convert_to_tensor=True)
        if self.memory_vectors is None:
            self.memory_vectors = new_vec
        else:
            self.memory_vectors = torch.cat((self.memory_vectors, new_vec), dim=0)
        self.memory_texts.append(text)

    def retrieve(self, query, k=2):
        if not self.memory_texts or self.memory_vectors is None:
            return []
        query_vec = self.model.encode([query], convert_to_tensor=True)
        cos_sim = torch.nn.functional.cosine_similarity(query_vec, self.memory_vectors)
        top_k_indices = torch.topk(cos_sim, min(k, len(self.memory_texts))).indices
        return [self.memory_texts[i] for i in top_k_indices]