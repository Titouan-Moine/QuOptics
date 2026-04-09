"""
Module de benchmarking pour le package InfoQ.

Ce module fournit des utilitaires et des abstractions pour mesurer les performances
de bouts de code ou de suites d'expériences reproducibles. Il est conçu pour
faciliter la collecte, l'agrégation et l'export des métriques de temps d'exécution
et d'usage des ressources (par ex. CPU, mémoire) dans le cadre de tests
comparatifs ou d'optimisation de code.

Fonctionnalités typiques :
- context managers et décorateurs pour mesurer facilement la durée d'exécution
    d'une fonction ou d'un bloc de code ;
- mécanismes de warmup et de répétitions pour réduire le bruit statistique ;
- agrégation des résultats (moyenne, médiane, écart-type, percentiles) ;
- exécution séquentielle ou parallèle d'une suite de benchmarks ;
- export des résultats au format CSV/JSON et génération de rapports sommaires ;
- options de configuration pour la précision des mesures et la reproductibilité.

Usage recommandé (exemple conceptuel) :
        with Timer() as t:
                ma_fonction()
        print(f"Durée : {t.elapsed:.6f}s")

Conception et extension :
Le module est pensé pour être extensible : on peut ajouter de nouveaux collecteurs
de métriques, formats d'export ou stratégies d'exécution sans modifier l'API
publique. Lors de l'ajout de mesures sensibles au système, privilégier la
répétition et le contrôle des conditions d'exécution (affinité CPU, caches, etc.).

Notes :
La précision réelle des mesures dépend de l'environnement d'exécution. Pour des
comparaisons fiables, exécuter les benchmarks sur des environnements stables et
documenter les conditions (versions, paramètres matériels, variables d'environnement).
"""
