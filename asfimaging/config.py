"""Load configuration paths independently of the process working directory."""

import json
from pathlib import Path


def load_config(filename):
    """Resolve all data/mapping paths against root, relative to the config file.

    Absolute paths remain absolute. Without ``root``, the config directory is
    the base. The example configs use ``root: ..`` for the repository root.
    """
    filename = Path(filename).expanduser().resolve()
    with filename.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"{filename}: the configuration must be a JSON object")
    root = Path(config.get("root", ".")).expanduser()
    if not root.is_absolute():
        root = filename.parent / root
    root = root.resolve()
    config["root"] = str(root)
    validate_config(config)  # before resolving paths, so messages show the values as written
    for section in ("data_paths", "mapping"):
        for key, value in config[section].items():
            path = Path(value).expanduser()
            config[section][key] = str((path if path.is_absolute() else root / path).resolve())
    return config


def require_files(config, *keys):
    """Report every missing input for a stage before starting expensive work."""
    missing = []
    for section, name in keys:
        value = config.get(section, {}).get(name)
        if not value or not Path(value).is_file():
            missing.append(f"  {section}.{name}: {value or '<not configured>'}")
    if missing:
        raise FileNotFoundError("Required input files are missing:\n" + "\n".join(missing))


# The synthetic pipeline (ODF library, Zeppelin kernel and the l-band features
# the forest is trained on) is fixed at this order; real-data features must
# use the same order so the feature vectors match.
SH_ORDER = 12


def check_sh_order(config):
    """Fail early when a config asks for an SH order the pipeline cannot use."""
    order = config.get('real_data_acquisition', {}).get('sh_order', SH_ORDER)
    if order != SH_ORDER:
        raise ValueError(
            f"real_data_acquisition.sh_order must be {SH_ORDER}: the synthetic "
            f"ODFs, Zeppelin kernel and trained features are fixed at order {SH_ORDER}"
        )
    return SH_ORDER


def results_directory(config, *parts, create=True):
    """Return <folder containing data_paths.dwi>/results[/parts...]."""
    directory = Path(config['data_paths']['dwi']).parent / 'results'
    directory = directory.joinpath(*parts)
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


# Optional mapping keys for outputs and their defaults under <dwi folder>/results.
OUTPUT_DIRECTORIES = {
    'features_dir': ('features',),
    'maps_dir': ('maps',),
    'slices_dir': (),
}


def output_directory(config, key, create=True):
    """mapping.<key> if configured, else its default under <dwi folder>/results."""
    configured = config.get('mapping', {}).get(key)
    if configured:
        directory = Path(configured)
    else:
        directory = results_directory(config, *OUTPUT_DIRECTORIES[key], create=False)
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def labels_path(config):
    """mapping.labels_path if configured, else <dwi folder>/results/RF_labels.npy."""
    configured = config.get('mapping', {}).get('labels_path')
    if configured:
        return Path(configured)
    return results_directory(config, create=False) / 'RF_labels.npy'


def metrics_path(config):
    """mapping.metrics_path if configured, else <model_path stem>_test_metrics.json."""
    configured = config.get('mapping', {}).get('metrics_path')
    if configured:
        return Path(configured)
    model = Path(config['mapping']['model_path'])
    return model.with_name(f'{model.stem}_test_metrics.json')


def n_jobs(config, section):
    """Parallel workers for a stage: real_data_preprocessing (real SH fitting),
    synthetic_data_parameters (simulation) or RF_model_parameters (forest)."""
    return int(config[section].get('n_jobs', 1))


def sh_fit_settings(config):
    """SH fit settings shared by real and synthetic data so features match."""
    preprocessing = config.get('real_data_preprocessing', {})
    return {
        'positivity': bool(preprocessing.get('enforce_non_negativity', True)),
        'lambda_reg': float(preprocessing.get('sh_lambda_reg', 0.0)),
    }


# ---------------------------------------------------------------------------
# Configuration schema and validation
# ---------------------------------------------------------------------------

class _Spec:
    """Expected type of a config value; `required` keys must be present."""

    def __init__(self, kind, required=True, choices=None):
        self.kind = kind          # 'number', 'int', 'bool', 'str', 'path', 'list', 'any'
        self.required = required
        self.choices = choices


def _req(kind, **kwargs):
    return _Spec(kind, True, **kwargs)


def _opt(kind, **kwargs):
    return _Spec(kind, False, **kwargs)


_FEATURE_FLAGS = ('zonal_ratios', 'sphm_high_b_lower', 'sphm_high_b_upper',
                  'sphvar_high_b_lower', 'sphvar_high_b_upper',
                  'spectra_high_b_lower', 'spectra_high_b_upper')

_BOUNDED = ('t2_intra', 't2_extra', 't2_gm', 'intra_dpar', 'intra_dperp',
            'extra_diff', 'csf_diso', 'gm_diso')

# Every key the code reads. Optional keys have defaults in the code.
CONFIG_SCHEMA = {
    'root': _opt('str'),
    'experiment': _opt('str'),
    'data_paths': {
        'dwi': _opt('path'), 'bvals': _opt('path'), 'bvecs': _opt('path'),
        'mask': _opt('path'), 'ODF': _opt('path'), 'scheme': _opt('path'),
    },
    'real_data_acquisition': {
        'high_bmin': _req('number'), 'high_bmax': _req('number'),
        'tolerance_bval': _req('number'), 'tolerance_b0': _req('number'),
        'TE': _req('number'), 'delta': _req('number'), 'Delta': _req('number'),
        'sh_order': _opt('int'),
    },
    'real_data_preprocessing': {
        'noise_distribution': _req('str', choices=('gaussian', 'rician', 'none')),
        'enforce_non_negativity': _opt('bool'),
        'sh_lambda_reg': _opt('number'),
        'n_jobs': _opt('int'),
    },
    'synthetic_data_parameters': {
        'N_signals': _req('int'), 'N_signals_test': _opt('int'),
        'random_seed_train': _opt('int'), 'random_seed_test': _opt('int'),
        'n_jobs': _opt('int'),
        'alphas_dirichlet': _req('list'), 'snr': _req('list'),
        **{f'{name}_{side}': _req('number') for name in _BOUNDED for side in ('lower', 'upper')},
        't2_dot': _req('number'), 't2_csf': _req('number'),
        'dot_diso': _req('number'), 'relative_vf_dot': _req('number'),
    },
    'features_labels': {
        **{flag: _req('bool') for flag in _FEATURE_FLAGS},
        # d_par and d_perp are always predicted; ASF optionally as a third target.
        'predicted_asf': _req('bool'),
    },
    'RF_model_parameters': {
        'n_estimators': _req('int'), 'max_depth': _req('any'),
        'min_samples_split': _req('any'), 'min_samples_leaf': _req('any'),
        'max_features': _req('any'), 'criterion': _req('str'),
        'random_state': _req('any'), 'n_jobs': _req('int'),
        'bootstrap': _opt('bool'), 'oob_score': _opt('bool'),
    },
    'mapping': {
        'synthetic_data': _req('path'), 'model_path': _req('path'),
        'scaler_path': _req('path'), 'labels_path': _opt('path'),
        'metrics_path': _opt('path'), 'features_dir': _opt('path'),
        'maps_dir': _opt('path'), 'slices_dir': _opt('path'),
    },
}


def _type_ok(value, kind):
    if kind == 'any':
        return True
    if kind == 'bool':
        return isinstance(value, bool)
    if kind == 'int':
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == 'number':
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind in ('str', 'path'):
        return isinstance(value, str) and bool(value.strip())
    if kind == 'list':
        return isinstance(value, list)
    raise ValueError(f"Unknown spec kind {kind!r}")


_KIND_NAMES = {'int': 'an integer', 'number': 'a number', 'bool': 'true or false',
               'str': 'a nonempty string', 'path': 'a nonempty path string', 'list': 'a list'}


def _check_keys(where, given, schema, errors):
    """Required/unknown keys and types for one mapping against its schema."""
    import difflib
    import warnings

    for key, spec in schema.items():
        if isinstance(spec, dict):
            if key not in given:
                errors.append(f"missing section '{key}'")
            elif not isinstance(given[key], dict):
                errors.append(f"'{key}' must be an object")
            else:
                _check_keys(f"{key}.", given[key], spec, errors)
            continue
        if key not in given:
            if spec.required:
                errors.append(f"missing {where}{key}")
            continue
        value = given[key]
        if not _type_ok(value, spec.kind):
            errors.append(f"{where}{key} must be {_KIND_NAMES[spec.kind]}, got {value!r}")
        elif spec.choices and value not in spec.choices:
            errors.append(f"{where}{key} must be one of {', '.join(spec.choices)}, got {value!r}")
    for key in given:
        if key not in schema:
            close = difflib.get_close_matches(key, list(schema), n=1)
            hint = f" (did you mean '{close[0]}'?)" if close else ''
            warnings.warn(f"Unknown config key '{where}{key}'{hint}; it is ignored.",
                          UserWarning, stacklevel=4)


def validate_config(config):
    """Check a configuration before any work starts.

    Raises ValueError listing every problem (missing keys, wrong types,
    impossible values) and warns about unknown keys, e.g. typos. Called by
    load_config; call it directly when building a config dict in Python.
    """
    errors = []
    _check_keys('', config, CONFIG_SCHEMA, errors)
    # Value checks run only for sections whose keys and types are correct,
    # so every problem is reported in one pass.
    bad = {name for name in CONFIG_SCHEMA if isinstance(CONFIG_SCHEMA[name], dict)
           and any(e.startswith((f"{name}.", f"missing {name}.", f"missing section '{name}'",
                                 f"'{name}'")) for e in errors)}

    def ok(*sections):
        return not bad.intersection(sections)

    acq = config.get('real_data_acquisition', {}) if ok('real_data_acquisition') else None
    if acq is not None:
        _check_acquisition(acq, errors)
    if ok('real_data_preprocessing') and config['real_data_preprocessing'].get('sh_lambda_reg', 0.0) < 0:
        errors.append("real_data_preprocessing.sh_lambda_reg cannot be negative")
    for section in ('real_data_preprocessing', 'synthetic_data_parameters', 'RF_model_parameters'):
        if ok(section) and config[section].get('n_jobs', 1) == 0:
            errors.append(f"{section}.n_jobs cannot be 0 (use a positive count or -1 for all cores)")
    if ok('synthetic_data_parameters'):
        _check_synthetic(config['synthetic_data_parameters'], errors)
    if ok('features_labels') and not any(config['features_labels'][f] for f in _FEATURE_FLAGS):
        errors.append("features_labels: select at least one feature family")
    if ok('RF_model_parameters'):
        rf = config['RF_model_parameters']
        if rf['n_estimators'] <= 0:
            errors.append("RF_model_parameters.n_estimators must be positive")
        if rf.get('oob_score', rf.get('bootstrap', True)) and not rf.get('bootstrap', True):
            errors.append("RF_model_parameters.oob_score requires bootstrap: true")

    if errors:
        raise ValueError("Invalid configuration:\n  " + "\n  ".join(errors))

    syn = config['synthetic_data_parameters']
    if syn['intra_dpar_lower'] < syn['intra_dperp_upper']:
        import warnings
        warnings.warn("synthetic_data_parameters: intra_dpar_lower < intra_dperp_upper, so some "
                      "simulated axons have d_par < d_perp; the ASF model assumes d_par >= d_perp.",
                      UserWarning, stacklevel=2)
    return config


def _check_acquisition(acq, errors):
    if not 0 < acq['high_bmin'] < acq['high_bmax']:
        errors.append(f"real_data_acquisition: need 0 < high_bmin < high_bmax, got "
                      f"{acq['high_bmin']:g} and {acq['high_bmax']:g}")
    for key in ('tolerance_bval', 'tolerance_b0', 'TE', 'delta', 'Delta'):
        if acq[key] <= 0:
            errors.append(f"real_data_acquisition.{key} must be positive")
    if acq['high_bmin'] < acq['high_bmax'] and \
            acq['tolerance_bval'] * 2 >= acq['high_bmax'] - acq['high_bmin']:
        errors.append("real_data_acquisition.tolerance_bval is so large that the two "
                      "high shells overlap")
    if acq['tolerance_b0'] >= acq['high_bmin'] - acq['tolerance_bval']:
        errors.append("real_data_acquisition.tolerance_b0 overlaps the high_bmin shell")
    if acq['Delta'] < acq['delta']:
        errors.append("real_data_acquisition: Delta must be at least delta")
    if acq.get('sh_order', SH_ORDER) != SH_ORDER:
        errors.append(f"real_data_acquisition.sh_order must be {SH_ORDER}: the synthetic "
                      f"ODFs, Zeppelin kernel and trained features are fixed at order {SH_ORDER}")



def _check_synthetic(syn, errors):
    for key in ('N_signals', 'N_signals_test'):
        if key in syn and syn[key] <= 0:
            errors.append(f"synthetic_data_parameters.{key} must be positive")
    for key in ('random_seed_train', 'random_seed_test'):
        if key in syn and syn[key] < 0:
            errors.append(f"synthetic_data_parameters.{key} cannot be negative")
    alphas = syn['alphas_dirichlet']
    if len(alphas) != 3 or not all(_type_ok(a, 'number') and a > 0 for a in alphas):
        errors.append("synthetic_data_parameters.alphas_dirichlet must be 3 positive numbers "
                      "(extra-axonal, GM, CSF)")
    snr = syn['snr']
    if len(snr) not in (2, 6) or not all(_type_ok(s, 'number') and s > 0 for s in snr):
        errors.append("synthetic_data_parameters.snr must be 2 positive numbers [low, high] or 6 "
                      "[wm_low, wm_high, gm_low, gm_high, csf_low, csf_high]")
    elif any(snr[i] > snr[i + 1] for i in range(0, len(snr), 2)):
        errors.append("synthetic_data_parameters.snr: each low value must not exceed its high value")
    for name in _BOUNDED:
        low, high = syn[f'{name}_lower'], syn[f'{name}_upper']
        if low < 0 or low > high:
            errors.append(f"synthetic_data_parameters: need 0 <= {name}_lower <= {name}_upper, "
                          f"got {low:g} and {high:g}")
    for key in ('t2_dot', 't2_csf', 't2_intra_lower', 't2_extra_lower', 't2_gm_lower'):
        if syn[key] <= 0:
            errors.append(f"synthetic_data_parameters.{key} must be positive")
    if syn['dot_diso'] < 0:
        errors.append("synthetic_data_parameters.dot_diso cannot be negative")
    if not 0 <= syn['relative_vf_dot'] <= 1:
        errors.append("synthetic_data_parameters.relative_vf_dot must be between 0 and 1")
