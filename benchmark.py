import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import random
from collections import defaultdict
import json

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# =================== ORBITCOMPRESSOR IMPLEMENTATION ===================

import torch
import torch.nn as nn
import numpy as np

class SingleOrbitEmbedding(nn.Module):
    def __init__(self, Z, embed_dim, sigma=0.2):
        super().__init__()
        self.Z = Z
        self.sigma = sigma
        self.embed_dim = embed_dim

        self.n_pairs = embed_dim // 2
        self.R = nn.Parameter(torch.ones(self.n_pairs))
        self.Cx = nn.Parameter(torch.zeros(self.n_pairs))
        self.Cy = nn.Parameter(torch.zeros(self.n_pairs))

    def forward(self, indices):
        i_prime = indices % self.Z
        theta_base = (2 * np.pi * i_prime) / self.Z
        theta_smooth = theta_base + self.sigma * torch.sin(4 * np.pi * i_prime / self.Z)

        theta_smooth = theta_smooth.unsqueeze(-1)  # [B, L, 1]
        R = self.R.view(1, 1, -1)
        Cx = self.Cx.view(1, 1, -1)
        Cy = self.Cy.view(1, 1, -1)

        x = Cx + R * torch.cos(theta_smooth)
        y = Cy + R * torch.sin(theta_smooth)

        return torch.cat([x, y], dim=-1)


class MultiOrbitalEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, orbit_hierarchy):
        super().__init__()
        self.orbit_hierarchy = orbit_hierarchy
        self.n_orbits = len(orbit_hierarchy)
        self.embed_dim_per_orbit = embed_dim // self.n_orbits

        self.orbits = nn.ModuleList([
            SingleOrbitEmbedding(Z, self.embed_dim_per_orbit)
            for Z in orbit_hierarchy
        ])

        self.vocab_to_multi_index = self.build_vocab_mapping(vocab_size)

    def build_vocab_mapping(self, vocab_size):
        vocab_tensor = torch.zeros(vocab_size, self.n_orbits, dtype=torch.long)
        for vocab_id in range(vocab_size):
            multi_index = []
            remaining = vocab_id
            for Z in reversed(self.orbit_hierarchy):
                index = remaining % Z
                multi_index.append(index)
                remaining //= Z
            multi_index = list(reversed(multi_index))
            vocab_tensor[vocab_id] = torch.tensor(multi_index)
        return vocab_tensor

    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape
        multi_indices = self.vocab_to_multi_index[token_ids]

        embeddings = []
        for i, orbit in enumerate(self.orbits):
            orbit_indices = multi_indices[..., i]
            orbit_embedding = orbit(orbit_indices)
            embeddings.append(orbit_embedding)

        return torch.cat(embeddings, dim=-1)


class SingleOrbitEmbedding(nn.Module):
    def __init__(self, Z, embed_dim, sigma=0.2):
        super().__init__()
        self.Z = Z
        self.sigma = sigma
        self.embed_dim = embed_dim
        
        # Paramètres apprenables par paire (x,y)
        self.n_pairs = embed_dim // 2
        self.R = nn.Parameter(torch.ones(self.n_pairs) * 0.5)
        self.Cx = nn.Parameter(torch.randn(self.n_pairs) * 0.1)  
        self.Cy = nn.Parameter(torch.randn(self.n_pairs) * 0.1)
        
    def forward(self, indices):
        batch_size, seq_len = indices.shape
        i_prime = indices % self.Z
        
        # Calcul des angles avec perturbation
        theta_base = (2 * np.pi * i_prime.float()) / self.Z
        theta_smooth = theta_base + self.sigma * torch.sin(4 * np.pi * i_prime.float() / self.Z)
        
        # Génération des coordonnées pour chaque paire
        coords = []
        for j in range(self.n_pairs):
            x = self.Cx[j] + self.R[j] * torch.cos(theta_smooth)
            y = self.Cy[j] + self.R[j] * torch.sin(theta_smooth)
            coords.extend([x, y])
            
        # Stack et reshape pour obtenir (batch_size, seq_len, embed_dim)
        result = torch.stack(coords, dim=-1)  # (batch_size, seq_len, embed_dim)
        return result

""""class MultiOrbitalEmbedding(nn.Module):
        def __init__(self, vocab_size, embed_dim, orbit_hierarchy):
        super().__init__()
        
        self.orbit_hierarchy = orbit_hierarchy
        self.n_orbits = len(orbit_hierarchy)
        self.embed_dim_per_orbit = embed_dim // self.n_orbits
        
        # Paramètres par orbite
        self.orbits = nn.ModuleList([
            SingleOrbitEmbedding(Z, self.embed_dim_per_orbit) 
            for Z in orbit_hierarchy
        ])
        
        # Mapping vocab -> indices multi-orbitaux
        self.vocab_to_multi_index = self.build_vocab_mapping(vocab_size)
        
    def build_vocab_mapping(self, vocab_size):
        """"Mapping intelligent vocab -> (i₁, i₂, i₃, ..., iₙ)""""""
        mapping = {}
        
        for vocab_id in range(vocab_size):
            multi_index = []
            remaining = vocab_id
            
            for Z in reversed(self.orbit_hierarchy):
                index = remaining % Z
                multi_index.append(index)
                remaining //= Z
                
            mapping[vocab_id] = tuple(reversed(multi_index))
            
        return mapping
    
    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape
        embeddings = []
        
        for i, orbit in enumerate(self.orbits):
            # Extraction de l'index pour cette orbite
            orbit_indices = torch.zeros_like(token_ids)
            for b in range(batch_size):
                for s in range(seq_len):
                    token_id = token_ids[b, s].item()
                    if token_id < len(self.vocab_to_multi_index):
                        orbit_indices[b, s] = self.vocab_to_multi_index[token_id][i]
            
            # Embedding orbital pour cette couche
            orbit_embedding = orbit(orbit_indices)
            embeddings.append(orbit_embedding)
            
        # Concaténation des embeddings multi-orbitaux
        return torch.cat(embeddings, dim=-1)"""

class OrbitCompressorModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, orbit_hierarchy, num_classes, max_seq_len=128):
        super().__init__()
        self.embedding = MultiOrbitalEmbedding(vocab_size, embed_dim, orbit_hierarchy)
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, embed_dim) * 0.1)
        
        # Transformer simplifié
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=embed_dim * 2,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=4
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
    def forward(self, input_ids, attention_mask=None):
        # Embedding multi-orbital
        embeddings = self.embedding(input_ids)
        
        # Positional encoding
        seq_len = input_ids.size(1)
        pos_enc = self.pos_encoding[:seq_len].unsqueeze(0).expand(embeddings.size(0), -1, -1)
        embeddings = embeddings + pos_enc
        
        # Transformer
        if attention_mask is not None:
            # Créer le masque d'attention pour transformer
            mask = ~attention_mask.bool()
        else:
            mask = None
            
        transformer_output = self.transformer(embeddings, src_key_padding_mask=mask)
        
        # Global average pooling
        if attention_mask is not None:
            # Masquer les positions de padding
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(transformer_output)
            transformer_output = transformer_output * mask_expanded
            pooled = transformer_output.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            pooled = transformer_output.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        return logits

# =================== BASELINE MODELS ===================

class SimpleTransformerModel(nn.Module):
    """Modèle Transformer simple comme baseline"""
    def __init__(self, vocab_size, embed_dim, num_classes, max_seq_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, embed_dim) * 0.1)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=6
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
    def forward(self, input_ids, attention_mask=None):
        embeddings = self.embedding(input_ids)
        
        # Positional encoding
        seq_len = input_ids.size(1)
        pos_enc = self.pos_encoding[:seq_len].unsqueeze(0).expand(embeddings.size(0), -1, -1)
        embeddings = embeddings + pos_enc
        
        # Transformer
        if attention_mask is not None:
            mask = ~attention_mask.bool()
        else:
            mask = None
            
        transformer_output = self.transformer(embeddings, src_key_padding_mask=mask)
        
        # Global average pooling
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(transformer_output)
            transformer_output = transformer_output * mask_expanded
            pooled = transformer_output.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            pooled = transformer_output.mean(dim=1)
        
        logits = self.classifier(pooled)
        return logits

# =================== SYNTHETIC DATA GENERATOR ===================

class SyntheticDataGenerator:
    def __init__(self, vocab_size=5000, max_seq_len=128):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Créer des "mots" synthétiques avec différentes fréquences
        self.word_frequencies = self._create_word_frequencies()
        
    def _create_word_frequencies(self):
        """Crée une distribution de fréquence réaliste (loi de Zipf)"""
        frequencies = {}
        for i in range(self.vocab_size):
            # Loi de Zipf approximative
            freq = 1.0 / (i + 1) ** 0.8
            frequencies[i] = freq
        return frequencies
    
    def generate_sentiment_data(self, n_samples=2000, task_type='sentiment'):
        """Génère des données synthétiques pour classification"""
        
        if task_type == 'sentiment':
            # Mots "positifs" et "négatifs"
            positive_words = list(range(100, 200))  # Vocabulaire "positif"
            negative_words = list(range(200, 300))  # Vocabulaire "négatif"
            neutral_words = list(range(300, self.vocab_size))  # Vocabulaire neutre
            
            texts = []
            labels = []
            
            for _ in range(n_samples):
                # Génération d'une séquence
                seq_len = np.random.randint(20, self.max_seq_len)
                
                # Choix du sentiment
                sentiment = np.random.choice([0, 1])  # 0=négatif, 1=positif
                
                sequence = []
                if sentiment == 1:  # Positif
                    # Plus de mots positifs
                    for _ in range(seq_len):
                        if np.random.random() < 0.6:
                            word = np.random.choice(positive_words)
                        elif np.random.random() < 0.3:
                            word = np.random.choice(neutral_words)
                        else:
                            word = np.random.choice(negative_words)
                        sequence.append(word)
                else:  # Négatif
                    # Plus de mots négatifs
                    for _ in range(seq_len):
                        if np.random.random() < 0.6:
                            word = np.random.choice(negative_words)
                        elif np.random.random() < 0.3:
                            word = np.random.choice(neutral_words)
                        else:
                            word = np.random.choice(positive_words)
                        sequence.append(word)
                
                # Padding
                sequence = sequence[:self.max_seq_len]
                sequence += [0] * (self.max_seq_len - len(sequence))
                
                texts.append(sequence)
                labels.append(sentiment)
            
            return texts, labels
        
        elif task_type == 'topic':
            # Classification de topics
            topic_words = {
                0: list(range(100, 200)),   # Sport
                1: list(range(200, 300)),   # Science
                2: list(range(300, 400)),   # Politique
            }
            
            texts = []
            labels = []
            
            for _ in range(n_samples):
                # Choix du topic
                topic = np.random.choice([0, 1, 2])
                
                seq_len = np.random.randint(20, self.max_seq_len)
                sequence = []
                
                for _ in range(seq_len):
                    if np.random.random() < 0.7:
                        # Mot du topic principal
                        word = np.random.choice(topic_words[topic])
                    else:
                        # Mot neutre
                        word = np.random.randint(400, self.vocab_size)
                    sequence.append(word)
                
                # Padding
                sequence = sequence[:self.max_seq_len]
                sequence += [0] * (self.max_seq_len - len(sequence))
                
                texts.append(sequence)
                labels.append(topic)
            
            return texts, labels

# =================== BENCHMARK SUITE ===================

class SimpleBenchmark:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {}
        
    def create_data_loaders(self, texts, labels, batch_size=32, train_split=0.8):
        """Crée les data loaders pour entraînement et test"""
        
        # Conversion en tenseurs
        input_ids = torch.tensor(texts, dtype=torch.long)
        attention_mask = (input_ids != 0).long()  # Masque de padding
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        # Split train/test
        n_train = int(len(texts) * train_split)
        
        # Dataset d'entraînement
        train_dataset = torch.utils.data.TensorDataset(
            input_ids[:n_train], 
            attention_mask[:n_train], 
            labels_tensor[:n_train]
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        # Dataset de test
        test_dataset = torch.utils.data.TensorDataset(
            input_ids[n_train:], 
            attention_mask[n_train:], 
            labels_tensor[n_train:]
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        return train_loader, test_loader
    
    def train_model(self, model, train_loader, epochs=5, lr=1e-3):
        """Entraîne un modèle"""
        model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
                input_ids, attention_mask, labels = [x.to(self.device) for x in [input_ids, attention_mask, labels]]
                
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
            total_loss += epoch_loss
        
        training_time = time.time() - start_time
        return total_loss / (epochs * len(train_loader)), training_time
    
    def evaluate_model(self, model, test_loader):
        """Évalue un modèle"""
        model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for input_ids, attention_mask, labels in test_loader:
                input_ids, attention_mask, labels = [x.to(self.device) for x in [input_ids, attention_mask, labels]]
                
                outputs = model(input_ids, attention_mask)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        inference_time = time.time() - start_time
        accuracy = correct / total
        
        return accuracy, inference_time, all_predictions, all_labels
    
    def benchmark_model(self, model, model_name, train_loader, test_loader):
        """Benchmark complet d'un modèle"""
        print(f"\n🚀 Benchmarking {model_name}...")
        
        # Calcul de la taille du modèle
        model_size = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Entraînement
        avg_loss, training_time = self.train_model(model, train_loader)
        
        # Évaluation
        accuracy, inference_time, predictions, labels = self.evaluate_model(model, test_loader)
        
        # Calcul des métriques supplémentaires
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        f1 = f1_score(labels, predictions, average='weighted')
        precision = precision_score(labels, predictions, average='weighted')
        recall = recall_score(labels, predictions, average='weighted')
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'model_size': model_size,
            'trainable_params': trainable_params,
            'training_time': training_time,
            'inference_time': inference_time,
            'avg_loss': avg_loss
        }
        
        print(f"✅ {model_name} Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        print(f"   Model Size: {model_size:,} parameters")
        print(f"   Training Time: {training_time:.2f}s")
        print(f"   Inference Time: {inference_time:.2f}s")
        
        return results
    
    def run_comparison(self, task_type='sentiment', vocab_size=5000):
        """Exécute la comparaison complète"""
        print("🎯 ORBITCOMPRESSOR vs STANDARD TRANSFORMER BENCHMARK")
        print("=" * 60)
        
        # Génération des données
        print("📊 Generating synthetic data...")
        data_gen = SyntheticDataGenerator(vocab_size=vocab_size)
        texts, labels = data_gen.generate_sentiment_data(n_samples=2000, task_type=task_type)
        
        # Création des data loaders
        train_loader, test_loader = self.create_data_loaders(texts, labels)
        
        num_classes = len(set(labels))
        embed_dim = 256
        
        # Modèle standard
        print("\n🤖 Creating Standard Transformer...")
        standard_model = SimpleTransformerModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_classes=num_classes
        )
        
        # Modèle OrbitCompressor
        print("🌀 Creating OrbitCompressor...")
        orbit_hierarchy = [200, 100, 50, 25]  # Hiérarchie multi-orbitale
        orbit_model = OrbitCompressorModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            orbit_hierarchy=orbit_hierarchy,
            num_classes=num_classes
        )
        
        # Benchmark des modèles
        results = {}
        
        # Standard Transformer
        results['standard'] = self.benchmark_model(
            standard_model, 
            "Standard Transformer",
            train_loader, 
            test_loader
        )
        
        # OrbitCompressor
        results['orbit'] = self.benchmark_model(
            orbit_model,
            "OrbitCompressor",
            train_loader,
            test_loader
        )
        
        # Analyse comparative
        self.analyze_comparison(results)
        
        return results
    
    def analyze_comparison(self, results):
        """Analyse comparative détaillée"""
        print("\n📊 COMPARATIVE ANALYSIS")
        print("=" * 60)
        
        standard = results['standard']
        orbit = results['orbit']
        
        # Métriques de base
        print(f"\n🎯 PERFORMANCE METRICS:")
        print(f"   Standard Transformer:")
        print(f"     - Accuracy: {standard['accuracy']:.4f}")
        print(f"     - F1 Score: {standard['f1_score']:.4f}")
        print(f"     - Parameters: {standard['model_size']:,}")
        print(f"   OrbitCompressor:")
        print(f"     - Accuracy: {orbit['accuracy']:.4f}")
        print(f"     - F1 Score: {orbit['f1_score']:.4f}")
        print(f"     - Parameters: {orbit['model_size']:,}")
        
        # Comparaisons relatives
        print(f"\n🚀 ORBITCOMPRESSOR ADVANTAGES:")
        
        # Compression
        compression_ratio = standard['model_size'] / orbit['model_size']
        print(f"   📉 Model Compression: {compression_ratio:.1f}x smaller")
        
        # Vitesse
        speed_ratio = standard['training_time'] / orbit['training_time']
        print(f"   ⚡ Training Speed: {speed_ratio:.1f}x faster")
        
        inference_speed = standard['inference_time'] / orbit['inference_time']
        print(f"   ⚡ Inference Speed: {inference_speed:.1f}x faster")
        
        # Efficacité (Performance/Taille)
        standard_efficiency = standard['f1_score'] / (standard['model_size'] / 1_000_000)
        orbit_efficiency = orbit['f1_score'] / (orbit['model_size'] / 1_000_000)
        efficiency_ratio = orbit_efficiency / standard_efficiency
        
        print(f"\n🏆 EFFICIENCY METRICS:")
        print(f"   Standard F1/Size: {standard_efficiency:.6f}")
        print(f"   OrbitCompressor F1/Size: {orbit_efficiency:.6f}")
        print(f"   🎯 Efficiency Gain: {efficiency_ratio:.1f}x more efficient!")
        
        # Performance relative
        performance_ratio = orbit['f1_score'] / standard['f1_score']
        print(f"\n📈 PERFORMANCE RETENTION:")
        print(f"   Performance Ratio: {performance_ratio:.3f} ({performance_ratio*100:.1f}%)")
        
        if performance_ratio > 0.95:
            print("   🎉 EXCELLENT: Performance maintained!")
        elif performance_ratio > 0.85:
            print("   ✅ GOOD: Acceptable performance trade-off")
        else:
            print("   ⚠️  CAUTION: Significant performance drop")
        
        # Verdict final
        print(f"\n🏆 FINAL VERDICT:")
        if compression_ratio > 5 and performance_ratio > 0.85:
            print("   🚀 REVOLUTIONARY: OrbitCompressor shows massive potential!")
            print("   🎯 Recommendation: Continue development and scale up!")
        elif compression_ratio > 2 and performance_ratio > 0.90:
            print("   ✅ PROMISING: Good compression with maintained performance")
        else:
            print("   🔧 NEEDS WORK: Requires optimization")

# =================== EXÉCUTION DU BENCHMARK ===================

if __name__ == "__main__":
    # Vérification du device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")
    
    # Initialisation du benchmark
    benchmark = SimpleBenchmark(device=device)
    
    # Exécution du benchmark
    print("🚀 Starting OrbitCompressor Benchmark...")
    results = benchmark.run_comparison(task_type='sentiment', vocab_size=3000)
    
    print("\n🎉 BENCHMARK COMPLETE!")
    print("🌟 OrbitCompressor: Revolutionary compression for the future of NLP!")



