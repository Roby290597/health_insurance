"""
conftest.py
Sorgt dafür, dass Tests unabhängig vom Aufrufort funktionieren:
main.py lädt Modell/Scaler über relative Pfade ("models/scaler.json"),
deshalb muss das Arbeitsverzeichnis beim Testlauf auf das Projekt-Root zeigen.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
