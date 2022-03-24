# Reconnaissance de plan - partie 1

Kaoutar BOUHAMIDI EL ALAOUI --- 21 157 917
Gaëtan LOUNES --- 21 175 773
Eliott THOMAS --- 21 164 874
Luca VAIO --- 21 154 698

## children

Cette fonction permet de retourner la liste des nœuds enfants d'un nœud passé en paramètre.

On ne retourne que les nœuds qui ne sont pas des obstacles et qui appartiennent à la carte bien sûr.

## heuristic

Cette fonction prend en paramètre le nœud de départ et le nœud d'arrivée et retourne la distance de Manhattan entre les deux points, autrement dit le coût du traget optimal.

## planner

On applique ici l'algorithme **A***. L'algorithme débute au nœud de départ, puis va explorer les autres nœuds de la carte. Nous stockons ces nœuds dans une file d'attente `PriorityQueue`, dont les nœuds explorés sont supprimés. Il a été choisi d'utiliser une file prioritaire plutôt qu'une simple liste pour des raisons de performances. Les nœuds ajoutés dans cette file d'attente sont triés automatiquement par coût (H+G) grâce à la surcharge de l'opérateur `__gt__`. Le prochain nœud observé sera donc celui dont le coût est le plus faible.

L'algorithme continue tant que cette file n'est pas vide (tant qu'il y a des nœuds à explorer). Si l'on atteint le nœud d'arrivée désiré, on reconstruit le chemin jusqu'au nœud de départ à l'aide des parents de chaque nœud.

Pour chaque nœud de la file d'attente on évalue le coût du chemin menant à chacun de ses enfants. Si l'enfant n'a jamais été observé (i.e. le nœud n'a pas de parent), on calcule le coût du chemin jusqu'au but passant par ce nœud. On lui assigne un parent (le nœud actuellement observé) et on l'ajoute à notre file d'attente. À l'inverse s'il a un parent, on regarde si le nouveau chemin est meilleur que l'ancien, auquel cas on met son coût à jour.

## predictMastersSardina

Cette fonction retourne une liste de probabilités correspondant à chaque but.

La première étape est d'obtenir deux plans : l'un entre la position de départ et le but, l'autre entre la position actuelle et le but.

Ensuite, la probabilité du but en question correspond à un softmax de la différence entre la longueur de ces deux plans.

## accuracy

Cette fonction correspond au calcul de la justesse des prédictions. Elle prend en paramètres les probabilités prédites par `predictMastersSardina` et la liste des buts corrects.
Comme précisé dans le sujet, une prédiction est correcte si la probabilité correspondant au but correct fait partie des probabilités maximales. Nous ne pouvons donc pas utiliser la méthode `argmax`, il faut itérer sur l'ensemble des probabilités.
