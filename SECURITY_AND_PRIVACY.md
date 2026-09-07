# Security and privacy

The release candidate excludes raw and preprocessed participant-level data, participant identifiers and mappings, participant-level labels and predictions, per-subject attributions, checkpoints, credentials, API keys, tokens, browser and cloud-drive configuration, caches, temporary files, and local logs. Public-candidate files are scanned for sensitive field names, email-like values outside the locked author contacts, credential patterns, and local absolute paths.

Checkpoints were inspected upstream with a safe weights-only load. No embedded personal path, subject-identifier field, credential, or personal-information finding was detected. All 30 checkpoint byte files remain excluded from the first public package by explicit author decision; their inventory and structural audit do not grant rights in the weights. Security concerns should be sent privately to the corresponding author without attaching sensitive data.
