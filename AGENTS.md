# Repository-Regeln

Diese Regeln gelten für das gesamte Repository.

## Code und README gemeinsam pflegen

- Prüfe bei jeder Codeänderung parallel, ob sie für `README.md` relevant ist.
- Ändert sich dokumentiertes oder für Nutzer relevantes Verhalten, passe
  `README.md` im selben Arbeitsschritt an. Das gilt insbesondere für
  Abhängigkeiten und Versionen, Installation, Startbefehle, Modelle,
  Standardwerte, API-Felder und Statuswerte sowie Fehlerbehandlung.
- Gleiche Beschreibungen und Beispiele mit dem tatsächlichen Code und der
  Konfiguration ab. Der implementierte Stand ist maßgeblich.
- Bei Änderungen ohne README-Relevanz ist keine rein formale Anpassung nötig.
- Prüfe vor Abschluss, dass Code und README weiterhin übereinstimmen und
  betroffene Beispiele aktuell sind.

## Python

- Verwende Python 3.12, vorzugsweise `.venv/bin/python3.12`, für Prüfungen und
  Projektbefehle.
