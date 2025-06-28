# 📋 Guide de l'Animateur - Workshop Apprentissage par Renforcement

## 🎯 Vue d'ensemble du Workshop

**Durée estimée :** 3-4 heures  
**Niveau :** Débutant à Intermédiaire en IA/ML  
**Prérequis :** Bases de Python, notions de machine learning optionnelles

## 📊 Structure Recommandée

### **Phase 1 : Introduction Théorique (30-45 min)**

#### Concepts à Présenter :
1. **Qu'est-ce que l'Apprentissage par Renforcement ?**
   - Agent, Environnement, Actions, États, Récompenses
   - Différence avec supervised/unsupervised learning
   - Exemples concrets : jeux vidéo, robots, voitures autonomes

2. **Deep Q-Learning (DQN)**
   - Problème : espaces d'états énormes 
   - Solution : approximation avec réseaux de neurones
   - Q-values : "qualité" d'une action dans un état donné

3. **Stratégie Epsilon-Greedy**
   - Dilemme exploration vs exploitation
   - Epsilon décroissant au fil du temps

4. **Équation de Bellman**
   - `Q(s,a) = reward + gamma * max(Q(s',a'))`
   - Récompense immédiate + potentiel futur

### **Phase 2 : Présentation du Projet (15 min)**

#### Démonstration :
- Montrer le code original fonctionnel
- Lancer l'entraînement pour quelques épisodes
- Expliquer l'architecture des fichiers
- Présenter les fichiers "workshop" à compléter

### **Phase 3 : Travaux Pratiques (2h30-3h)**

## 🔄 Ordre d'Exécution des Exercices

### **EXERCICE 1 : État (45 min) - DÉBUTANT** 
**Pourquoi commencer par là :** C'est la base conceptuelle

#### **Points d'Attention :**
- Bien faire comprendre que l'agent ne "voit" que des nombres
- Les 11 éléments booléens : 3 dangers + 4 directions + 4 positions nourriture
- Aide commune : confusion entre directions absolues et relatives

#### **Indices à Donner Progressivement :**
1. "Commencez par récupérer `head = game.snake[0]`"
2. "Les points potentiels sont à 20 pixels de distance (taille d'un bloc)"
3. "Pour les dangers, pensez aux directions relatives : droit devant, à droite, à gauche"

#### **Erreurs Fréquentes :**
- Oublier `np.array(state, dtype=int)` en retour
- Mal gérer les directions relatives vs absolues
- État avec mauvais nombre d'éléments

---

### **EXERCICE 4 : Réseau de Neurones (30 min) - INTERMÉDIAIRE**
**Pourquoi après l'état :** L'architecture est plus simple à comprendre

#### **Points d'Attention :**
- ReLU pour la couche cachée, pas d'activation pour la sortie
- Importance du nombre de neurones : entrée=11, sortie=3

#### **Indices à Donner :**
1. "`tf.keras.layers.Dense(hidden_size, activation='relu')`"
2. "Dans `call()`, c'est juste `x = self.dense1(input)` puis `return self.dense2(x)`"

#### **Erreurs Fréquentes :**
- Activation sur la couche de sortie
- Oublier le `return` dans `call()`

---

### **EXERCICE 2 : Actions Epsilon-Greedy (45 min) - INTERMÉDIAIRE**
**Pourquoi maintenant :** Nécessite la compréhension de l'état et du modèle

#### **Points d'Attention :**
- Epsilon décroît : beaucoup d'exploration au début, moins après
- `final_move` : vecteur one-hot [1,0,0], [0,1,0], ou [0,0,1]
- `np.argmax()` pour trouver la meilleure action prédite

#### **Algorithme à Expliquer au Tableau :**
```
SI random < epsilon :
    action = aléatoire
SINON :
    prédictions = modèle(état)
    action = argmax(prédictions)
```

#### **Erreurs Fréquentes :**
- Ne pas réinitialiser `final_move = [0,0,0]`
- Confondre `randint(0,2)` et `randint(0,3)`
- Mauvais formatage de l'état pour le modèle

---

### **EXERCICE 3 : Équation de Bellman (45 min) - AVANCÉ**
**Pourquoi en dernier :** Le plus conceptuellement difficile

#### **Points d'Attention :**
- C'est le cœur théorique de l'apprentissage par renforcement
- Distinction cas terminal vs non-terminal
- `np.argmax(action[idx])` trouve quelle action a été prise

#### **Explication Détaillée Nécessaire :**
1. **Si épisode terminé :** `Q_new = reward` (pas de futur)
2. **Si épisode continue :** `Q_new = reward + gamma * max(Q_next)`
3. **Mise à jour selective :** seule la Q-value de l'action prise est modifiée

#### **Erreurs Fréquentes :**
- Oublier le cas `if not done[idx]`
- `np.max(new_pred[idx])` vs `np.max(new_pred)`
- `np.argmax(action[idx])` pour trouver l'index de l'action

## 🆘 Gestion des Difficultés

### **Si les Participants sont Bloqués :**
1. **Premier niveau :** Hints progressifs dans les commentaires
2. **Deuxième niveau :** Montrer la solution d'une partie spécifique
3. **Dernier recours :** Fichiers solutions complets dans `/solutions/`

### **Si le Code ne Marche Pas :**
- Vérifier que tous les imports sont corrects dans les fichiers workshop
- S'assurer que `pygame` et `tensorflow` sont installés
- Les `pass` doivent être remplacés par du code

### **Adaptations selon le Niveau :**

#### **Groupe Débutant :**
- Plus de temps sur la théorie
- Faire l'Exercice 1 ensemble au tableau
- Simplifier l'Exercice 3 (donner plus d'indices)

#### **Groupe Avancé :**
- Moins de temps sur la théorie
- Défis bonus : modifier les récompenses, l'architecture réseau
- Discussion sur autres algorithmes RL (A3C, PPO, etc.)

## 🎮 Phase de Test et Observation

### **Après Implémentation Complète :**
1. **Lancer l'entraînement ensemble**
2. **Observer les premières parties :** mouvement aléatoire
3. **Après 50-100 parties :** premiers apprentissages
4. **Après 200+ parties :** stratégies optimales

### **Points à Faire Remarquer :**
- Epsilon qui diminue → moins d'exploration
- Score moyen qui augmente
- Stratégies émergentes (longer les murs, spirales, etc.)

## 💡 Questions Fréquentes des Participants

### **"Pourquoi le serpent ne s'améliore pas ?"**
- Vérifier que l'équation de Bellman est correcte
- L'apprentissage prend du temps (100-500 parties)
- Epsilon trop élevé = trop d'exploration

### **"À quoi sert gamma ?"**
- Contrôle l'importance du futur vs présent
- Gamma = 0.9 → 90% d'importance pour les récompenses futures
- Gamma proche de 1 → vision long terme

### **"Pourquoi replay buffer ?"**
- Éviter la corrélation entre expériences consécutives
- Utiliser efficacement toutes les expériences passées
- Stabiliser l'apprentissage

## 🏆 Conclusion et Ouverture (15 min)

### **Récapitulatif des Concepts :**
- Représentation d'état
- Epsilon-greedy
- Équation de Bellman
- Deep Q-Learning

### **Extensions Possibles :**
- Autres jeux (Pong, Atari, etc.)
- Autres algorithmes (Policy Gradient, Actor-Critic)
- Applications réelles (robotique, finance, etc.)

### **Ressources pour Aller Plus Loin :**
- Cours en ligne (Coursera, edX)
- Bibliothèques (OpenAI Gym, Stable-Baselines3)
- Papiers de recherche (DeepMind, OpenAI)

---

**Bon Workshop ! 🚀** 