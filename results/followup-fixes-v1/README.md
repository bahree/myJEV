# Follow-up verification evidence

These receipts describe the source and results identified in `completion.json`; later documentation edits do not revise that historical run.

`public-replay.json` records an in-memory export of private commit `4dbe6c4f481b2003d2b6b46e47f21531fbef4cd8`. Its `public_snapshot_sha256` identifies that intermediate export, not a committed public `public-snapshot.json`. Public readers can reproduce the thirteen named outputs from public commit `b342476adde69ecffc69e958cde4ffaf4500fbd0`, which includes the same regenerators and result inputs. The `source_script_sha256` and `output_sha256` entries identify the reproducible files; the private commit is not required to obtain them.

The public tree at `dcd5d7ae35af7857eb32d83f3abde57f6bec49e1` adds receipts to that snapshot. Neither a receipt-only commit nor a card-only release changes the model weights or calibration.
