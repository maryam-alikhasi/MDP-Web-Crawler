# MDP-Web-Crawler
This project was developed as part of the **Fundamentals and Applications of Artificial Intelligence** course at the University of Isfahan.  
It implements **Markov Decision Process (MDP)** algorithms — **Value Iteration** and **Policy Iteration** — to analyze navigation through a set of HTML pages with rewards and links.

---

## Project Summary

- **Language**: Python 3  
- **Core Idea**: Model web navigation as an MDP where each HTML page is a state, links are actions, and meta tags define terminal rewards.  
- **Algorithms Implemented**:
  - `Value Iteration`
  - `Policy Iteration`  
- **Input**: A directory of `.html` files containing links and optional reward metadata.  
- **Output**:  
  - State value estimates `V(s)`  
  - Optimal greedy policy for each page  

---

##  Features

- **HTML Parsing**:  
  Extracts pages, links (`<a href>`), and terminal rewards (`<meta name="reward" …>`).  

- **MDP Construction**:  
  Builds transition probabilities and reward functions from parsed corpus.  

- **Value Iteration**:  
  Iteratively updates state values until convergence using the Bellman optimality equation.  

- **Policy Iteration**:  
  Alternates between policy evaluation and policy improvement until the policy stabilizes.  

- **Driver Program**:  
  Runs chosen algorithm and prints:
  - **State Values (V)**
  - **Optimal Policy (best action per page)**

---

## Usage

Run the project with a directory of HTML corpus:

```bash
python MDP_Project.py <corpus_directory>
```

Example:

```bash
python MDP_Project.py corpus0
```

## Learning Outcomes

Understanding MDP formulation from real-world data (web navigation).
Comparing Value Iteration vs Policy Iteration in terms of convergence and results.
Practical experience with reward design and transition modeling in AI systems.