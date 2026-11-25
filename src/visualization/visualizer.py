import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, silhouette_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.svm import SVC
import pennylane as qml

from src.models.hybrid_model import EnhancedHybridModel


class MLVisualizer:
    """Comprehensive plotting and metrics visualization suite."""

    def __init__(self, out_dir: str = '.', fmt: str = 'png'):
        self.out_dir = out_dir
        self.fmt = fmt
        os.makedirs(out_dir, exist_ok=True)

        self.batch_size = 64
        self.num_qubits = 4
        self.num_layers = 2
        self.metrics_history = {
            'Hybrid': {
                'train_loss': [], 'val_loss': [], 'accuracy': [],
                'precision': [], 'recall': [], 'f1': [], 'robustness': [],
                'time': [], 'cpu': [], 'memory': [],
            },
            'Classical': {
                'train_loss': [], 'val_loss': [], 'accuracy': [],
                'precision': [], 'recall': [], 'f1': [], 'robustness': [],
                'time': [], 'cpu': [], 'memory': [],
            },
            'Hybrid_Test': {
                'val_loss': [], 'accuracy': [], 'precision': [],
                'recall': [], 'f1': [],
            },
            'Classical_Test': {
                'val_loss': [], 'accuracy': [], 'precision': [],
                'recall': [], 'f1': [],
            },
        }

    def _save(self, name: str):
        filepath = os.path.join(self.out_dir, f"{name}.{self.fmt}")
        plt.savefig(filepath, format=self.fmt, bbox_inches='tight', dpi=300)
        plt.close()

    def update_metrics(self, model_type: str, metrics_dict: dict):
        if model_type not in self.metrics_history:
            return
        for key, val in metrics_dict.items():
            if key in self.metrics_history[model_type]:
                self.metrics_history[model_type][key].append(val)

    def plot_training_curves(self, base_filename: str = 'training_curves'):
        plt.figure(figsize=(8, 6))
        plt.plot(self.metrics_history['Hybrid']['train_loss'], 'b-', label='Hybrid Train')
        plt.plot(self.metrics_history['Hybrid']['val_loss'], 'b--', label='Hybrid Val')
        plt.plot(self.metrics_history['Classical']['train_loss'], 'r-', label='Classical Train')
        plt.plot(self.metrics_history['Classical']['val_loss'], 'r--', label='Classical Val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        self._save(f'{base_filename}_loss')

        plt.figure(figsize=(8, 6))
        plt.plot(self.metrics_history['Hybrid']['accuracy'], 'g-', label='Hybrid')
        plt.plot(self.metrics_history['Classical']['accuracy'], 'm-', label='Classical')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Validation Accuracy')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        self._save(f'{base_filename}_accuracy')

        plt.figure(figsize=(8, 6))
        plt.plot(self.metrics_history['Hybrid']['f1'], 'c-', label='Hybrid')
        plt.plot(self.metrics_history['Classical']['f1'], 'y-', label='Classical')
        plt.xlabel('Epoch')
        plt.ylabel('Macro F1 Score')
        plt.title('F1 Score Progression')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        self._save(f'{base_filename}_f1')

        if any(self.metrics_history['Hybrid']['robustness']):
            plt.figure(figsize=(8, 6))
            x_steps = range(5, len(self.metrics_history['Hybrid']['robustness']) * 5 + 1, 5)
            plt.plot(x_steps, self.metrics_history['Hybrid']['robustness'], 'k-', label='Hybrid')
            plt.plot(x_steps, self.metrics_history['Classical']['robustness'], 'k--', label='Classical')
            plt.xlabel('Epoch')
            plt.ylabel('Robustness (%)')
            plt.title('Adversarial Robustness (PGD)')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            self._save(f'{base_filename}_robustness')

    def plot_resource_usage(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.metrics_history['Hybrid']['time'], 'b-', label='Hybrid')
        plt.plot(self.metrics_history['Classical']['time'], 'r-', label='Classical')
        plt.title('Training Duration per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Seconds')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        self._save('resource_time')

        plt.figure(figsize=(8, 6))
        plt.plot(self.metrics_history['Hybrid']['cpu'], 'b-', label='Hybrid')
        plt.plot(self.metrics_history['Classical']['cpu'], 'r-', label='Classical')
        plt.title('CPU Utilization per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('CPU Usage (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        self._save('resource_cpu')

        plt.figure(figsize=(8, 6))
        plt.plot(self.metrics_history['Hybrid']['memory'], 'b-', label='Hybrid')
        plt.plot(self.metrics_history['Classical']['memory'], 'r-', label='Classical')
        plt.title('Memory Footprint per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Memory (GB)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        self._save('resource_memory')

    def plot_confusion_matrix(self, model, data_loader, device, class_names, model_name: str):
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix — {model_name}')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        self._save(f'confusion_matrix_{model_name}')

    def plot_class_distribution(self, dataset, title: str):
        targets = dataset.targets if hasattr(dataset, 'targets') else [y for _, y in dataset]
        plt.figure(figsize=(12, 8))
        sns.countplot(x=targets)
        plt.title(f'Class Distribution — {title}')
        plt.xlabel('Class Label')
        plt.ylabel('Sample Count')
        self._save('class_distribution')

    def plot_feature_space(self, model, data_loader, device, model_name: str, n_samples: int = 1000):
        model.eval()
        features = []
        labels = []

        with torch.no_grad():
            for i, (data, target) in enumerate(data_loader):
                if i * self.batch_size >= n_samples:
                    break
                data = data.to(device)

                if isinstance(model, EnhancedHybridModel):
                    x = model.classical_net(data)
                    x = x.view(x.size(0), -1)
                    x = model.feature_reducer(x)
                    x = model.qlayer(x)
                else:
                    x = model.conv_net(data)
                    x = x.view(x.size(0), -1)
                    if hasattr(model, 'head'):
                        for layer in list(model.head.children())[:-1]:
                            x = layer(x)

                features.append(x.cpu().numpy())
                labels.append(target.cpu().numpy())

        features = np.concatenate(features)[:n_samples]
        labels = np.concatenate(labels)[:n_samples]

        self._plot_pca(features, labels, model_name)
        self._plot_tsne(features, labels, model_name)
        self._plot_decision_boundaries(features, labels, model_name)

    def _plot_pca(self, features, labels, model_name: str):
        plt.figure(figsize=(12, 8))
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features)
        self._plot_decision_surface(features_pca, labels)

        scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels,
                              cmap='tab10', alpha=0.6, edgecolors='w', s=40)
        plt.title(f'{model_name} Feature Space — PCA\n'
                  f'Explained Variance: {pca.explained_variance_ratio_.sum():.2f}')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.colorbar(scatter, label='Class')
        plt.grid(True)
        plt.tight_layout()
        self._save(f'feature_space_pca_{model_name}')

    def _plot_tsne(self, features, labels, model_name: str):
        plt.figure(figsize=(12, 8))
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        features_tsne = tsne.fit_transform(features)
        self._plot_decision_surface(features_tsne, labels)

        scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels,
                              cmap='tab10', alpha=0.6, edgecolors='w', s=40)
        plt.title(f'{model_name} Feature Space — t-SNE')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.colorbar(scatter, label='Class')
        plt.grid(True)
        plt.tight_layout()
        self._save(f'feature_space_tsne_{model_name}')

    def _plot_decision_surface(self, X, y):
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        svm = SVC(kernel='rbf', gamma=2)
        svm.fit(X, y)
        Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.1, cmap='tab10')
        plt.contour(xx, yy, Z, colors='k', linewidths=0.5, alpha=0.5)

    def _plot_decision_boundaries(self, features, labels, model_name: str):
        plt.figure(figsize=(12, 8))
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features)
        self._plot_decision_surface(features_pca, labels)

        scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels,
                              cmap='tab10', alpha=0.8, edgecolors='w', s=40)

        lda = LinearDiscriminantAnalysis(n_components=1)
        lda.fit(features, labels)
        separation_score = lda.score(features, labels)
        sil_score = silhouette_score(features, labels)

        plt.title(f'{model_name} Class Separation\n'
                  f'LDA Separation: {separation_score:.3f} | Silhouette Score: {sil_score:.3f}')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.colorbar(scatter, label='Class')
        plt.grid(True)
        plt.tight_layout()
        self._save(f'class_separation_{model_name}')

    def plot_quantum_circuit(self, quantum_circuit, weights):
        qubits = getattr(self, 'num_qubits', weights.shape[1] if hasattr(weights, 'shape') else 4)
        layers = getattr(self, 'num_layers', weights.shape[0] if hasattr(weights, 'shape') else 2)

        dummy_input = torch.zeros(2 ** qubits)
        dummy_weights = torch.randn((layers, qubits))

        fig, ax = qml.draw_mpl(quantum_circuit)(dummy_input, dummy_weights)
        plt.title('Quantum Circuit Architecture')
        self._save('quantum_circuit')

    def plot_metric_comparison(self):
        metrics = ['accuracy', 'f1', 'robustness', 'time']
        hybrid_metrics = []
        classical_metrics = []

        for m in metrics:
            if m in self.metrics_history['Hybrid'] and self.metrics_history['Hybrid'][m]:
                hybrid_metrics.append(np.mean(self.metrics_history['Hybrid'][m]))
            if m in self.metrics_history['Classical'] and self.metrics_history['Classical'][m]:
                classical_metrics.append(np.mean(self.metrics_history['Classical'][m]))

        x = np.arange(len(metrics))
        width = 0.35

        plt.figure(figsize=(12, 6))
        plt.bar(x - width / 2, hybrid_metrics, width, label='Hybrid')
        plt.bar(x + width / 2, classical_metrics, width, label='Classical')

        plt.title('Model Performance Comparison')
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(True, axis='y')
        self._save('metric_comparison')

    def plot_test_results(self):
        if not self.metrics_history['Hybrid_Test']['accuracy']:
            return

        metrics = ['accuracy', 'precision', 'recall', 'f1']
        hybrid_metrics = [self.metrics_history['Hybrid_Test'][m][-1] for m in metrics]
        classical_metrics = [self.metrics_history['Classical_Test'][m][-1] for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        plt.figure(figsize=(12, 6))
        plt.bar(x - width / 2, hybrid_metrics, width, label='Hybrid')
        plt.bar(x + width / 2, classical_metrics, width, label='Classical')

        plt.title('Test Set Evaluation')
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(True, axis='y')
        self._save('test_results')

    def plot_sample_images(self, data_loader, title: str, n_images: int = 10):
        data_iter = iter(data_loader)
        images, labels = next(data_iter)

        plt.figure(figsize=(15, 3))
        for i in range(min(n_images, len(images))):
            plt.subplot(1, n_images, i + 1)
            if images[i].shape[0] == 1:
                plt.imshow(images[i].squeeze(), cmap='gray')
            else:
                plt.imshow(np.transpose(images[i].numpy(), (1, 2, 0)))
            plt.title(f"Label: {labels[i].item()}")
            plt.axis('off')
        plt.suptitle(title)
        plt.tight_layout()
        self._save(f'sample_images_{title.lower().replace(" ", "_")}')

    def _plot_predictions(self, model, data_loader, device, model_name: str, n_images: int = 10):
        model.eval()
        data_iter = iter(data_loader)
        images, labels = next(data_iter)
        images, labels = images[:n_images].to(device), labels[:n_images].to(device)

        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

        plt.figure(figsize=(15, 3))
        for i in range(min(n_images, len(images))):
            plt.subplot(1, n_images, i + 1)
            if images[i].shape[0] == 1:
                plt.imshow(images[i].cpu().squeeze(), cmap='gray')
            else:
                plt.imshow(np.transpose(images[i].cpu().numpy(), (1, 2, 0)))

            color = 'green' if preds[i] == labels[i] else 'red'
            plt.title(f"T: {labels[i].item()}\nP: {preds[i].item()}", color=color)
            plt.axis('off')

        plt.suptitle(f'Sample Predictions — {model_name}')
        plt.tight_layout()
        self._save(f'predictions_{model_name}')
