# NDT homework 2024-25: preliminary design of a fuel pin

Homework for *Nuclear Design and Technology* (Politecnico di Milano, A.Y. 2024-25): preliminary design and
verification of a MOX fuel pin with austenitic steel cladding for a test channel of a sodium-cooled fast reactor,
over one 360-day cycle at 38.7 kW/m. The assignment is [`Homework_2024-2025.pdf`](Homework_2024-2025.pdf).

- `code/`: Python model of the pin (`main.py`, `functions.py`). It computes the axial power profile, the coolant
  and cladding temperatures, fuel restructuring and Pu redistribution, swelling, the gap evolution,
  fission gas release and plenum pressure. It writes plots to `code/plots/` and results to CSV files.
- `notes/`: derivations and working notes (`diario_di_bordo.tex/.pdf`).

```bash
cd code && python main.py
```
