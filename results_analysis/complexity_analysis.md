# Complexity Analysis of the Best Models and Their Ensemble

Time complexity of training and inference of the best models (svm, RoBERTa, and Llama-3 ft) and their ensemble (response to comment R1.C8).

| Model | Training ($n$ comments) | Inference (one comment) | Ensemble | Δ F1 |
|---|---|---|---|---|
| svm with Gpt emb | $O(n \cdot m \cdot P_{emb} + n^2 \cdot d)$ | $O(m \cdot P_{emb} + s \cdot d)$ | $\approx P_{L} / P_{emb} \approx 27$ | 11.2% (0.663 → 0.737) |
| RoBERTa | $O(E \cdot n \cdot m \cdot P_{R})$ | $O(m \cdot P_{R})$ | $\approx P_{L} / P_{R} \approx 64$ | 6.5% (0.692 → 0.737) |
| Llama-3 ft | $O(E \cdot n \cdot m \cdot P_{L})$ | $O(m \cdot P_{L})$ | $\approx 1$ | 2.9% (0.716 → 0.737) |
| Ensemble | $\approx O(E \cdot n \cdot m \cdot P_{L})$ | $\approx O(m \cdot P_{L})$ | – | – |

- $n$: number of training comments
- $m$: maximum number of tokens of a comment
- $E$: number of epochs
- $P$: number of parameters of a model, i.e., $P_{emb} \approx$ 300M for the Gpt emb model (unofficial estimate, since OpenAI does not disclose this information), $P_{R}$ = 125M for RoBERTa, and $P_{L}$ = 8B for Llama-3
- $d$: dimension of the embeddings
- $s$: number of support vectors of svm
- Ensemble: number of times more operations per comment the ensemble requires compared with the model
- Δ F1: F1 improvement of the ensemble over the model

The training and inference costs of the models grow linearly with their number of parameters, and the cost of the ensemble is the sum of the costs of its models. Hence, the cost of the ensemble is dominated by Llama-3 ft (8B parameters), and the cost of svm is dominated by the Gpt emb model (about 300M parameters). Compared with Llama-3 ft, the ensemble has a similar cost and improves the F1 score by 2.9%, although it requires maintaining three models. Compared with RoBERTa and svm, it requires about 64 and 27 times more operations per comment for F1 improvements of 6.5% and 11.2%, respectively.
