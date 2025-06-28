# 🐍 Workshop : Apprentissage par Renforcement avec Snake Game

Bienvenue dans ce workshop d'introduction à l'apprentissage par renforcement ! Vous allez apprendre les concepts clés en implémentant un agent qui apprend à jouer au jeu Snake.

## 🎯 Objectifs Pédagogiques

À la fin de ce workshop, vous comprendrez :
- Les concepts fondamentaux de l'apprentissage par renforcement
- La représentation d'état dans un environnement de jeu
- La stratégie epsilon-greedy (exploration vs exploitation)
- L'équation de Bellman et le Q-Learning
- L'architecture des réseaux de neurones pour l'IA

## 🏗️ Structure du Projet

```
├── game.py              # ✅ Le jeu Snake (complet)
├── helper.py            # ✅ Visualisation des scores (complet)
├── agent_workshop.py    # 🔧 Agent à compléter (EXERCICES 1-2)
├── trainer_workshop.py  # 🔧 Entraîneur à compléter (EXERCICE 3)
├── model_workshop.py    # 🔧 Réseau de neurones à compléter (EXERCICE 4)
└── solutions/           # 💡 Solutions complètes (à consulter si bloqué)
```

## 📚 Exercices à Réaliser

### 🥉 **EXERCICE 1 : Représentation de l'État** (Débutant)
**Fichier :** `agent_workshop.py` → fonction `getState()`

**Mission :** Créer une représentation numérique de l'état du jeu.

**Concepts :** L'agent doit "voir" son environnement sous forme de données numériques.

**À implémenter :**
- Détecter les dangers (collisions potentielles)
- Identifier la direction actuelle du serpent
- Localiser la nourriture par rapport à la tête

---

### 🥈 **EXERCICE 2 : Stratégie Epsilon-Greedy** (Intermédiaire)
**Fichier :** `agent_workshop.py` → fonction `getAction()`

**Mission :** Équilibrer exploration (actions aléatoires) et exploitation (utiliser le modèle).

**Concepts :** Comment un agent apprend-il de nouvelles stratégies tout en optimisant celles qu'il connaît ?

**À implémenter :**
- Calcul d'epsilon décroissant
- Décision exploration vs exploitation
- Utilisation du modèle pour prédire les meilleures actions

---

### 🥇 **EXERCICE 3 : Équation de Bellman** (Avancé)
**Fichier :** `trainer_workshop.py` → fonction `trainer()`

**Mission :** Implémenter le cœur théorique de l'apprentissage par renforcement.

**Concepts :** Comment évaluer la qualité d'une action en combinant récompense immédiate et potentiel futur ?

**À implémenter :**
- L'équation : `Q(s,a) = reward + gamma * max(Q(s',a'))`
- Gestion des épisodes terminés
- Mise à jour des Q-values

---

### 🏆 **EXERCICE 4 : Architecture du Réseau** (Intermédiaire)
**Fichier :** `model_workshop.py` → classe `QNetwork`

**Mission :** Créer un réseau de neurones pour approximer la fonction Q.

**Concepts :** Comment transformer un état de jeu en évaluation d'actions ?

**À implémenter :**
- Couches denses avec TensorFlow
- Fonction d'activation ReLU
- Propagation avant des données

## 🚀 Comment Démarrer

### 1. **Installation des Dépendances**
```bash
pip install tensorflow numpy pygame matplotlib
```

### 2. **Ordre Recommandé des Exercices**
1. Commencez par l'**Exercice 1** (État) - fondamental
2. Passez à l'**Exercice 4** (Réseau) - architecture de base  
3. Continuez avec l'**Exercice 2** (Action) - logique de décision
4. Terminez par l'**Exercice 3** (Bellman) - le plus complexe

### 3. **Test de Votre Code**
```bash
python agent_workshop.py
```

## 💡 Conseils pour Réussir

### 🔍 **Débogage**
- Utilisez `print()` pour vérifier vos valeurs
- L'état doit avoir 11 éléments booléens (True/False)
- Les actions sont des listes [1,0,0], [0,1,0], ou [0,0,1]

### 📖 **Ressources d'Aide**
- Lisez attentivement les commentaires dans chaque exercice
- Les indices sont fournis étape par étape
- Consultez les fichiers originaux (`agent.py`, etc.) si vraiment bloqué

### 🎮 **Observation de l'Entraînement**
- Au début : le serpent bouge aléatoirement (exploration)
- Progressivement : il apprend à éviter les murs
- Finalement : il développe des stratégies pour attraper la nourriture

## 🧠 Concepts Clés à Retenir

### **Apprentissage par Renforcement**
- Agent, environnement, actions, états, récompenses
- Objectif : maximiser les récompenses cumulées

### **Deep Q-Learning**
- Utilise un réseau de neurones pour approximer Q(état, action)
- Mémorise les expériences passées (replay buffer)
- Stratégie epsilon-greedy pour l'exploration

### **Équation de Bellman**
- Relie la valeur d'une action à la récompense immédiate + potentiel futur
- Facteur gamma : importance des récompenses futures
- Base théorique de tous les algorithmes de Q-Learning

## 🏁 Résultats Attendus

Après implémentation correcte :
- Le serpent apprend à éviter les murs et son propre corps
- Le score moyen augmente progressivement
- L'agent développe des stratégies optimales

## 📞 Aide et Support

- **Erreurs de syntaxe :** Vérifiez l'indentation Python
- **Erreurs de logique :** Relisez les indices dans les commentaires
- **Performances :** L'apprentissage peut prendre 100-500 parties
- **Questions :** N'hésitez pas à demander de l'aide !

---

**🎉 Bon Workshop et Bon Apprentissage !**

*Rappelez-vous : L'apprentissage par renforcement imite notre façon d'apprendre - par essais, erreurs, et amélioration continue !* 