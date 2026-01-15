# Paper artifact CSV schema (recommended)

## cross_dataset_summary.csv
Columns:
- dataset_id,institution,data_version
- n_artifacts,n_mission,n_course,n_syllabus,n_program,n_outcome_report
- has_outcomes,n_outcomes_rows
- embedding,lenses,objectives,constraints
- pareto_size,feasible_rate
- run_id,report_dir
- data_hash,code_hash,generated_at

## cross_dataset_outcome_sanity.csv
Columns:
- dataset_id,institution,year_range
- n_units_total,n_units_with_outcomes,coverage
- method,embedding,lenses,market_targets
- spearman_mean,spearman_std
- mae_mean,mae_std
- raw_hash_outcomes,data_hash,code_hash,generated_at
