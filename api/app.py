import gradio as gr
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── CHARGEMENT DU MODÈLE ──────────────────────────────────────────────────────
print("Chargement du modèle RoBERTa...")
tokenizer = AutoTokenizer.from_pretrained('./models/roberta')
model     = AutoModelForSequenceClassification.from_pretrained('./models/roberta')
model.eval()
print("✅ Modèle chargé")


# ── FONCTION DE PRÉDICTION ────────────────────────────────────────────────────
def predict(tweet):
    if not tweet.strip():
        return "⚠️ Veuillez entrer un tweet", {}, ""

    inputs = tokenizer(tweet, truncation=True, padding=True,
                       max_length=128, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=-1).squeeze().numpy()

    prob_non_disaster = float(probs[0])
    prob_disaster     = float(probs[1])
    prediction        = np.argmax(probs)

    if prediction == 1:
        label = "🚨 DISASTER — Ce tweet décrit probablement une catastrophe"
        confidence = prob_disaster
    else:
        label = "💬 NON-DISASTER — Ce tweet n'est probablement pas lié à une catastrophe"
        confidence = prob_non_disaster

    probs_dict = {
        '🚨 Disaster'     : prob_disaster,
        '💬 Non-disaster' : prob_non_disaster
    }

    confidence_text = f"Confiance : **{confidence*100:.1f}%**"
    return label, probs_dict, confidence_text


# ── FONCTION D'EXPLICATION ────────────────────────────────────────────────────
def explain_prediction(tweet):
    if not tweet.strip():
        return "⚠️ Veuillez entrer un tweet", ""

    inputs = tokenizer(tweet, truncation=True, padding=True,
                       max_length=128, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)
        base_probs = torch.softmax(outputs.logits, dim=-1).squeeze().numpy()

    base_pred = int(np.argmax(base_probs))
    base_score = float(base_probs[base_pred])

    words = tweet.split()
    importances = []

    for i, word in enumerate(words):
        modified_tweet = ' '.join(words[:i] + words[i+1:])
        if not modified_tweet.strip():
            importances.append(0.0)
            continue

        modified_inputs = tokenizer(modified_tweet, truncation=True, padding=True,
                                     max_length=128, return_tensors='pt')

        with torch.no_grad():
            modified_outputs = model(**modified_inputs)
            modified_probs = torch.softmax(modified_outputs.logits, dim=-1).squeeze().numpy()

        importance = float(base_probs[base_pred] - modified_probs[base_pred])
        importances.append(importance)

    if max(abs(i) for i in importances) > 0:
        max_imp = max(abs(i) for i in importances)
        normalized = [i / max_imp for i in importances]
    else:
        normalized = importances

    html_words = []
    for word, imp in zip(words, normalized):
        if imp > 0:
            color = '#27AE60' if base_pred == 1 else '#3498DB'
            intensity = min(abs(imp), 1.0)
            html_words.append(
                f'<span style="background-color: {color}{int(intensity*150):02x}; '
                f'padding: 2px 6px; margin: 2px; border-radius: 4px; '
                f'font-weight: bold;">{word}</span>'
            )
        elif imp < -0.01:
            html_words.append(
                f'<span style="background-color: #BDC3C7; padding: 2px 6px; '
                f'margin: 2px; border-radius: 4px;">{word}</span>'
            )
        else:
            html_words.append(f'<span style="padding: 2px 6px; margin: 2px;">{word}</span>')

    highlighted_text = '<div style="font-size: 1.2rem; line-height: 2;">' + ' '.join(html_words) + '</div>'

    word_importance = sorted(zip(words, importances), key=lambda x: abs(x[1]), reverse=True)[:5]

    explanation_text = f"""
### 🔍 Analyse des mots importants

**Prédiction** : {'🚨 Disaster' if base_pred == 1 else '💬 Non-disaster'} (confiance : {base_score*100:.1f}%)

**Méthode** : Leave-One-Out — on retire chaque mot un par un et on observe
comment la prédiction change.

**TOP 5 des mots les plus influents :**
"""
    for i, (word, imp) in enumerate(word_importance, 1):
        sign = '✅' if imp > 0 else '❌'
        explanation_text += f"\n{i}. {sign} **{word}** (impact: {imp:+.3f})"

    explanation_text += """

**Légende :**
- 🟢 / 🔵 : mots qui ont **renforcé** la prédiction
- ⬜ : mots qui ont **affaibli** la prédiction
- ⚪ : mots neutres
"""

    return highlighted_text, explanation_text


# ── INTERFACE GRADIO ──────────────────────────────────────────────────────────
with gr.Blocks(
    title="Disaster Tweets Classifier",
    theme=gr.themes.Soft(primary_hue="red", secondary_hue="blue")
) as demo:

    gr.Markdown("""
    # 🚨 Disaster Tweets Classifier

    ### Détection automatique de tweets décrivant des catastrophes réelles

    Cette application utilise **RoBERTa fine-tuné** (F1 = 0.771) sur le dataset
    Kaggle Disaster Tweets pour distinguer les tweets décrivant de vraies catastrophes
    des tweets non liés à des catastrophes.
    """)

    with gr.Tabs():
        # ── ONGLET 1 : PRÉDICTION ─────────────────────────────────────────────
        with gr.Tab("🎯 Prédiction"):
            gr.Markdown("## Saisissez un tweet pour obtenir une prédiction")

            with gr.Row():
                with gr.Column(scale=2):
                    tweet_input = gr.Textbox(
                        label="Tweet",
                        placeholder="Ex: A massive earthquake hit California today...",
                        lines=3
                    )
                    predict_btn = gr.Button("🔮 Prédire", variant="primary", size="lg")

                with gr.Column(scale=1):
                    prediction_label = gr.Textbox(label="Résultat", interactive=False)
                    probs_output = gr.Label(label="Probabilités", num_top_classes=2)
                    confidence_md = gr.Markdown()

            gr.Markdown("### 💡 Exemples à tester")
            gr.Examples(
                examples=[
                    ["Just felt an earthquake! Buildings shaking in downtown LA."],
                    ["My heart is on fire for this new movie"],
                    ["BREAKING: Massive flood hits the city, residents evacuating"],
                    ["This pizza is the bomb!"],
                    ["Wildfire spreading rapidly through California forests"],
                    ["I'm dying of laughter watching this comedy"]
                ],
                inputs=[tweet_input]
            )

            predict_btn.click(predict, inputs=[tweet_input],
                              outputs=[prediction_label, probs_output, confidence_md])

        # ── ONGLET 2 : EXPLICATION ────────────────────────────────────────────
        with gr.Tab("🔍 Explication"):
            gr.Markdown("""
            ## Pourquoi le modèle prédit-il cela ?

            Cette page utilise la méthode **Leave-One-Out** pour identifier les mots
            qui ont le plus contribué à la décision.
            """)

            with gr.Row():
                with gr.Column():
                    tweet_input_exp = gr.Textbox(
                        label="Tweet à expliquer",
                        placeholder="Ex: Massive fire destroys building in downtown...",
                        lines=3
                    )
                    explain_btn = gr.Button("🔍 Expliquer", variant="primary", size="lg")

            gr.Markdown("### 🎨 Tweet annoté")
            highlighted_output = gr.HTML(label="Mots influents")

            gr.Markdown("### 📊 Analyse détaillée")
            explanation_output = gr.Markdown()

            explain_btn.click(explain_prediction, inputs=[tweet_input_exp],
                              outputs=[highlighted_output, explanation_output])

        # ── ONGLET 3 : À PROPOS ───────────────────────────────────────────────
        with gr.Tab("ℹ️ À propos"):
            gr.Markdown("""
            ## À propos du projet

            ### Contexte
            En cas de catastrophe naturelle, les médias sociaux comme Twitter deviennent
            une source précieuse d'information en temps réel. Cette application aide à
            distinguer automatiquement les vrais tweets de catastrophe des tweets non-pertinents.

            ### Modèle utilisé
            **RoBERTa** (Robustly Optimized BERT Pretraining Approach) fine-tuné sur
            le dataset Kaggle Disaster Tweets :
            - **F1 classe 1** : 0.771
            - **Recall** : 0.814
            - **Precision** : 0.732
            - **Accuracy** : 0.910

            ### Approche
            30 modèles ont été testés et comparés dans ce projet :
            - TF-IDF Classique (Naive Bayes, LogReg, SVM)
            - Embeddings (Word2Vec, GloVe Twitter)
            - Ensemblistes (Random Forest, XGBoost, Voting, Stacking)
            - Deep Learning (RoBERTa, BERTweet, DistilBERT, BiLSTM, TextCNN)

            ### Limitations
            - Anglais uniquement
            - Tweets de 2020 (peut être moins performant sur du vocabulaire récent)

            ---
            **Auteur** : Aïssatou Samb
            **Cours** : Machine Learning 2 — ISE2 2026
            """)

if __name__ == "__main__":
    demo.launch()