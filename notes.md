Visualisierung der Ergebnisse
Auswertungsmethode überlegen
Virtual Machine besorgen
evtl andere Clustering-Algorithmen
Docker?

Look into Co-Word Analysis (might be transferrable to result visualization)
https://cran.r-project.org/web/packages/bibliometrix/vignettes/bibliometrix-vignette.html

  * multi-dimensional scaling
  * correspondence analysis
  * multiple correspondence analysis
  * TSNE (t-distributed stochastic neighbor embedding)

Gilt die Dreiecksungleichung?
Auf wie vielen Dimensionen arbeiten wir wirklich (1 oder n)?
Anwendungsgebiete finden
Code profilen und optimieren (Bottlenecks?)

Neue Idee:
  Graph mit Ergebnislisten als Knoten und ungerichtete Kanten mit Gewicht RBO -> Pruning und schauen, ob der Graph an einem bestimmten Punkt in unzusammenhängende Subgraphen "zerfällt"
