#!/usr/bin/env python3
"""CSV-driven UWB simulator that mirrors the UWBTag API.

This simulator reads two CSVs (anchors and positions) and plays them back
in time order. When the position file reaches EOF the simulator sets
anchors and position to None to indicate no more data.

Usage: SimUWB(anchors_csv=None, positions_csv=None)
"""
import csv
import time
import threading
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple, cast

from Robot.subsystems.KalmanStateEstimator import KalmanStateEstimator

logger = logging.getLogger(__name__)


@dataclass
class Position:
    x: float
    y: float
    z: float
    quality: int
    timestamp: float


class SimUWB:
    """Singleton-like simulator for UWB tag data from CSV files."""

    _instance = None

    def __new__(cls, anchors_csv: Optional[str] = None, positions_csv: Optional[str] = None, interval: float = 0.01):
        if cls._instance is None:
            cls._instance = super(SimUWB, cls).__new__(cls)
            cls._instance._init(anchors_csv, positions_csv, interval)
        return cls._instance

    def _init(self, anchors_csv: Optional[str], positions_csv: Optional[str], interval: float):
        # CSV paths (relative to repository by default)
        import os
        root = os.path.dirname(__file__)
        sim_files = os.path.join(root, 'sim_files')

        self.anchors_csv = anchors_csv or os.path.join(sim_files, 'uwb_anchors.csv')
        # prefer a simulation positions file in sim_files; if not present fall back to example_output
        self.positions_csv = positions_csv or os.path.join(sim_files, 'uwb_positions.csv')
        if not os.path.exists(self.positions_csv):
            # try example_output location (useful for repository examples)
            example_path = os.path.join(root, '..', '..', 'example_output')
            # do a simple search for a file named uwb_positions.csv in example_output
            for dirpath, _, files in os.walk(example_path):
                if 'uwb_positions.csv' in files:
                    self.positions_csv = os.path.join(dirpath, 'uwb_positions.csv')
                    break

        self.interval = interval

        # internal state
        self.is_connected = False
        self.is_reading = False
        self.read_thread: Optional[threading.Thread] = None
        self.position_lock = threading.RLock()

        # tag_info mirrors real UWBTag usage
        self.tag_info = {'anchors': None, 'position': None}

        self._anchors_timeline = []  # list of (timestamp, anchors_list)
        self._positions_timeline = []  # list of dicts rows
        self._pos_index = 0

        self.state_estimator = KalmanStateEstimator()

        # load CSVs if available
        try:
            self._load_anchors()
        except Exception:
            logger.debug('No anchors CSV loaded or parse error')

        try:
            self._load_positions()
        except Exception:
            logger.debug('No positions CSV loaded or parse error')

    def _load_anchors(self):
        anchors = {}
        with open(self.anchors_csv, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # row fields: timestamp,port,name,id,x,y,z,range
                try:
                    ts = float(row.get('timestamp') or 0.0)
                except Exception:
                    ts = time.time()

                name = (row.get('name') or '').strip()
                anchor_id = (row.get('id') or '').strip()
                try:
                    x = float(row.get('x')) if row.get('x') not in (None, '') else None
                    y = float(row.get('y')) if row.get('y') not in (None, '') else None
                    z = float(row.get('z')) if row.get('z') not in (None, '') else None
                    rng = float(row.get('range')) if row.get('range') not in (None, '') else None
                except Exception:
                    x = y = z = rng = None

                if anchor_id:
                    anchors.setdefault(ts, [])
                    if x is not None and y is not None and z is not None:
                        anchors[ts].append({'name': name or anchor_id, 'id': anchor_id, 'position': (x, y, z), 'range': rng or 0.0})

        # convert anchors dict to a sorted timeline
        items = sorted(anchors.items(), key=lambda x: x[0])
        self._anchors_timeline = items  # [(timestamp, [anchor,...]), ...]

        # keep a merged latest anchors list for quick access as well
        if items:
            # use the last timestamp's anchors as baseline
            self._latest_anchors = items[-1][1].copy()
        else:
            self._latest_anchors = None

    def _load_positions(self):
        # positions CSV columns like: timestamp,x1,y1,z1,quality1,x2,y2,z2,quality2,...
        rows = []
        with open(self.positions_csv, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = float(row.get('timestamp') or 0.0)
                except Exception:
                    continue

                rows.append({'timestamp': ts, 'row': row})

        # sort by timestamp
        rows.sort(key=lambda r: r['timestamp'])
        self._positions_timeline = rows

    def connect(self) -> bool:
        """Pretend to open a connection for the simulator."""
        self.is_connected = True
        logger.info('SimUWB connected (CSV playback)')
        return True

    def disconnect(self):
        """Disconnect and stop any playback thread."""
        try:
            self.stop_reading()
        except Exception:
            pass

        self.is_connected = False
        with self.position_lock:
            self.tag_info['anchors'] = None
            self.tag_info['position'] = None

        logger.info('SimUWB disconnected')

    def get_location_data(self) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Position]]:
        """Return the current anchors list and Position (or None, None)."""
        with self.position_lock:
            anchors = self.tag_info['anchors']
            pos = self.tag_info['position']
            # return shallow copy of anchors like real UWBTag
            if anchors is None:
                return None, None
            return [a.copy() for a in anchors], pos

    def start_continuous_reading(self, interval: float = None, debug: bool = False):
        if self.is_reading:
            logger.warning('SimUWB already reading')
            return

        self.is_reading = True
        if interval is not None:
            self.interval = interval

        def read_loop():
            # play through positions timeline; when finished, set values to None
            n = len(self._positions_timeline)
            if n == 0:
                logger.warning('No positions loaded for SimUWB')
                # nothing to play; mark outputs None and stop
                with self.position_lock:
                    self.tag_info['anchors'] = None
                    self.tag_info['position'] = None
                self.is_reading = False
                return

            idx = 0
            while self.is_reading and idx < n:
                entry = self._positions_timeline[idx]
                ts = entry['timestamp']
                row = entry['row']

                # build anchors for this timestamp: prefer nearest anchors timeline entry not later than ts
                anchors = None
                if self._anchors_timeline:
                    # find last anchors with timestamp <= ts
                    anchors = None
                    for a_ts, a_list in self._anchors_timeline:
                        if a_ts <= ts:
                            anchors = a_list
                        else:
                            break

                # pick first available position block in the row
                pos = None
                # fields come in groups (x1,y1,z1,quality1),(x2,y2,...)
                # iterate groups
                # find how many groups by parsing keys
                keys = [k for k in row.keys() if k.startswith('x') or k.startswith('y') or k.startswith('z') or k.startswith('quality')]
                # simpler: iterate group index starting at 1
                for g in range(1, 10):
                    try:
                        xs = row.get(f'x{g}', '')
                        ys = row.get(f'y{g}', '')
                        zs = row.get(f'z{g}', '')
                        qs = row.get(f'quality{g}', '')
                        if xs not in (None, '') and ys not in (None, '') and zs not in (None, ''):
                            try:
                                x = float(xs)
                                y = float(ys)
                                z = float(zs)
                                q = int(float(qs)) if qs not in (None, '') and qs != '' else 0
                                pos = Position(x=x, y=y, z=z, quality=q, timestamp=ts)
                                break
                            except Exception:
                                continue
                    except Exception:
                        break

                # update tag_info
                with self.position_lock:
                    self.tag_info['anchors'] = anchors.copy() if anchors is not None else None
                    self.tag_info['position'] = pos

                # if we have a valid position, feed the EKF like real device
                if pos is not None:
                    try:
                        tag_pos_meas = __import__('numpy').array([pos.x, pos.y, pos.z], dtype=float)
                        try:
                            self.state_estimator.update_uwb_range(tag_pos_meas)
                        except Exception:
                            logger.debug('EKF update skipped in SimUWB')
                    except Exception:
                        pass

                # advance index and sleep according to desired interval
                idx += 1
                # compute wait from next timestamp if available, otherwise use self.interval
                if idx < n:
                    next_ts = self._positions_timeline[idx]['timestamp']
                    dt = max(0.0, min(1.0, next_ts - ts))
                    time.sleep(max(self.interval, dt))
                else:
                    # EOF reached
                    break

            # At EOF set None values and stop reading
            with self.position_lock:
                self.tag_info['anchors'] = None
                self.tag_info['position'] = None

            self.is_reading = False
            logger.info('SimUWB playback finished (EOF reached)')

        self.read_thread = threading.Thread(target=read_loop, daemon=True)
        self.read_thread.start()
        logger.info('SimUWB started continuous reading')

    def stop_reading(self):
        self.is_reading = False
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join()
        logger.info('SimUWB stopped reading')

    def get_latest_position(self) -> Optional[Position]:
        with self.position_lock:
            return self.tag_info['position']

    def get_latest_anchor_info(self) -> Optional[List[Dict[str, Any]]]:
        with self.position_lock:
            anchors = cast(Optional[List[Dict[str, Any]]], self.tag_info.get('anchors'))
            if anchors is None:
                return None
            return [a.copy() for a in anchors]


__all__ = ['SimUWB', 'Position']
