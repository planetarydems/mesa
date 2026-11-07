"""Final decisions for exact proposals, separate from historical visual reviews."""
import hashlib
import json
from pathlib import Path


def load_decisions(root):
    pointer = Path(root) / 'data/DISPOSITIONS_LATEST.json'
    if not pointer.exists():
        return {}, None
    pin = json.loads(pointer.read_text())
    path = Path(root) / pin['path']
    if hashlib.sha256(path.read_bytes()).hexdigest() != pin['sha256']:
        raise ValueError('campaign decision manifest changed')
    manifest = json.loads(path.read_text())
    decisions = {}
    for row in manifest['sites']:
        case = row['case_id']
        if case in decisions:
            raise ValueError('duplicate campaign decision: ' + case)
        if row['decision'] not in {'accept', 'reject', 'pending_processing'}:
            raise ValueError('invalid final decision: ' + case)
        if row['decision'] == 'pending_processing':
            dep = row.get('processing_dependency', {})
            if not all(dep.get(k) for k in ('job_id', 'expected_output', 'question')):
                raise ValueError('pending decision without concrete processing: ' + case)
        decisions[case] = row
    return decisions, pin


def permits_proposal(review, decisions, pending_resolutions=None):
    row = decisions.get(review['case_id'])
    if row is None:
        return True
    if row['array_sha256'] != review['array_sha256']:
        raise ValueError('new candidate needs a versioned final decision: ' + review['case_id'])
    resolved = (pending_resolutions or {}).get(review['case_id'])
    if resolved is not None:
        if row['decision'] != 'pending_processing' or resolved['array_sha256'] != review['array_sha256'] or resolved['new_review_sha256'] != review.get('review_file_sha256') or review['decision'] != 'accept_local_morphology':
            raise ValueError('invalid pending-only proposal transition')
        return True
    if row['source_visual_decision'] != review['decision']:
        raise ValueError('changed review needs a versioned final decision: ' + review['case_id'])
    return row['decision'] == 'accept'


def load_pending_resolution(path, decisions, decision_pin, reviews):
    """One explicit completed edge dependency; never reopens a closed reject."""
    from edge_certificate import CASE, ARRAY, sha256
    if path is None:
        return {}
    proof = json.loads(Path(path).read_text())
    if proof.get('schema_version') != 1 or proof.get('prior_decision_manifest') != decision_pin:
        raise ValueError('pending resolution prior ledger mismatch')
    if proof.get('case_id') != CASE or proof.get('array_sha256') != ARRAY:
        raise ValueError('pending resolution outside supported exact case')
    old = decisions.get(CASE, {})
    if old.get('decision') != 'pending_processing' or old.get('array_sha256') != ARRAY:
        raise ValueError('pending resolution cannot reopen closed decision')
    if proof.get('prior_processing_dependency') != old.get('processing_dependency'):
        raise ValueError('pending resolution dependency changed')
    done = proof.get('completed_processing', {})
    if done.get('job_id') != '6156348' or done.get('job_id') != old['processing_dependency']['job_id'] or done.get('status') != 'COMPLETED 0:0':
        raise ValueError('pending dependency is not this completed job')
    if done.get('path') != old['processing_dependency']['expected_output'] or sha256(done.get('verification_path', done['path'])) != done.get('sha256'):
        raise ValueError('pending resolution output binding mismatch')
    result = json.loads(Path(done.get('verification_path', done['path'])).read_text())
    if result.get('validation_pass') is not True or result.get('case_id') != CASE or result.get('array_sha256') != ARRAY:
        raise ValueError('pending resolution completed evidence failed')
    review = reviews.get(CASE, {})
    new = proof.get('new_review', {})
    if review.get('decision') != 'accept_local_morphology' or review.get('array_sha256') != ARRAY or new.get('path') != review.get('review_file') or new.get('sha256') != review.get('review_file_sha256') or sha256(new['path']) != new['sha256']:
        raise ValueError('pending resolution Sol review binding mismatch')
    return {CASE: {'array_sha256': ARRAY, 'new_review_sha256': new['sha256'], 'proof_path': str(path), 'proof_sha256': sha256(path), 'completed_job_id': '6156348'}}
