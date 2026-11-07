"""One publisher for individual and union releases, including direct CLI calls."""
from contextlib import contextmanager
import fcntl
from pathlib import Path


@contextmanager
def publication_lock(stage):
    stage = Path(stage)
    stage.mkdir(parents=True, exist_ok=True)
    with (stage / '.publication.lock').open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def union_member_cases(patch):
    return [m['case_id'] for m in patch.get('members', [])] if patch.get('kind') == 'same_candidate_union' else []


def union_review_concerns(patch, reviews, union_reviews=None):
    """A later withdrawal of a component approval suspends the entire union."""
    concerns = []
    if patch.get('kind') == 'same_candidate_union':
        uid = patch['case_id'];current = (union_reviews or {}).get(uid)
        if current is None:
            raise ValueError('incomplete cumulative union review set: ' + uid)
        if current.get('decision') != 'accept_local_morphology' or current.get('array_sha256') != patch['review']['array_sha256']:
            concerns.append('union_approval:' + uid)
        required = {r['site'] for r in patch['union_nearby_sites'] if r['weight'] > 0}
        cleared = {int(r['site']) for r in current.get('neighbor_clearances', []) if r['decision'] == 'safe_in_this_footprint'}
        if not required.issubset(cleared):concerns.append('union_neighbours:' + uid)
    for member in patch.get('members', []):
        case = member['case_id']
        if case not in reviews:
            raise ValueError('incomplete cumulative review set for union member: ' + case)
        review = reviews[case]
        if review['decision'] != 'accept_local_morphology' or review['array_sha256'] != member['array_sha256']:
            concerns.append(case)
    return concerns
