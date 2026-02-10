# Archived Notebooks

These notebooks are **deprecated** and scheduled for removal in v2.0.

## Archive Policy

| Notebook | Deprecated | Reason | Replacement |
|----------|------------|--------|-------------|
| `step4_integration_test.ipynb` | 2026-02-09 | Tested Instant4D pipeline (replaced by De3DGS) | `../step4_de3dgs_validation.ipynb` |
| `integration_test_sam2.ipynb` | 2026-02-09 | Tested Stage 2 SAM-2 (stage removed) | N/A (Stage 2 no longer exists) |

## Important Notes

- **Do not run these notebooks** - dependencies may be broken
- Use `notebooks/step4_de3dgs_validation.ipynb` for GPU validation instead
- These files are preserved for historical reference and rollback capability
- Scheduled for permanent removal in the v2.0 release

## Why Archive Instead of Delete?

1. **Rollback capability**: If De3DGS has issues, we can reference the old approach
2. **Knowledge transfer**: New team members can understand the migration history
3. **Debugging**: If legacy issues arise, the original code is available
4. **Compliance**: Clear deprecation trail for audit purposes

## Migration Context

The brain-dance project migrated from Instant4D to Deformable 3D Gaussians (De3DGS) in February 2026.

Key changes:
- Stage 2 (SAM-2 object segmentation) was **removed** - De3DGS handles dynamics implicitly
- Stage 3 adapter changed from `Instant4DAdapter` to `Deformable3DGSAdapter`
- Training time increased from 2-5 min to 10-30 min (but quality improved significantly)

See `docs/migration/deformable-3dgs-migration.md` for full migration details.
