import phonenumbers
from phonenumbers import timezone as pn_timezone
from datetime import datetime, timedelta, time
import pytz
from typing import List, Set, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict
import csv
import argparse
import sys
from typing import Optional
import logging
from pathlib import Path
import heapq

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class CallMetrics:
    """Store call duration metrics"""
    agent_call_duration: int
    agent_wait_duration: int

    @property
    def total_duration(self) -> int:
        return self.agent_call_duration + self.agent_wait_duration

@dataclass
class PhoneCall:
    """Represents a phone call to be made"""
    number: str
    timezones: Set[str]
    processed: bool = False


class PhoneNumberProcessor:
    """Handle phone number parsing and timezone detection"""

    def parse_number(self, phone_str: str) -> phonenumbers.PhoneNumber:
        """Parse phone string into PhoneNumber object"""
        try:
            return phonenumbers.parse(phone_str, "US")
        except phonenumbers.NumberParseException:
            raise ValueError(f"Invalid phone number format: {phone_str}")

    def get_timezones(self, phone_number: phonenumbers.PhoneNumber) -> Set[str]:
        """Get all possible timezones for a phone number"""
        return set(pn_timezone.time_zones_for_number(phone_number))


class OptimizedCallScheduler:
    """Handles the optimized scheduling of calls"""

    def __init__(self, start_hour: int = 9, end_hour: int = 20):
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.processor = PhoneNumberProcessor()

    def group_by_timezone(self, phone_numbers: List[str]) -> Dict[frozenset, List[PhoneCall]]:
        """Group phone numbers by their timezone combinations"""
        timezone_groups = defaultdict(list)

        for number in phone_numbers:
            phone_obj = self.processor.parse_number(number)
            timezones = frozenset(self.processor.get_timezones(phone_obj))
            timezone_groups[timezones].append(PhoneCall(number, timezones))

        return timezone_groups

    def get_callable_window(self, timezones: Set[str], current_time: datetime) -> Tuple[datetime, datetime]:
        """Get the next available calling window for a set of timezones, using a timezone cache."""

        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=pytz.UTC)

        earliest_start = None
        latest_end = None

        # Timezone cache (dictionary to store timezone objects)
        if not hasattr(self, "_timezone_cache"):  # Initialize cache only once
            self._timezone_cache = {}

        for tz_name in timezones:
            tz = self._timezone_cache.get(tz_name)
            if tz is None:
                tz = pytz.timezone(tz_name)
                self._timezone_cache[tz_name] = tz  # Store in cache

            local_time = current_time.astimezone(tz)

            if local_time.hour < self.start_hour:
                start = local_time.replace(hour=self.start_hour, minute=0, second=0, microsecond=0)
            elif local_time.hour >= self.end_hour:
                start = (local_time.replace(hour=self.start_hour, minute=0, second=0, microsecond=0) + timedelta(days=1))
            else:
                start = local_time.replace(minute=0, second=0, microsecond=0)

            end = local_time.replace(hour=self.end_hour, minute=0, second=0, microsecond=0)
            if start.date() > local_time.date():
                end += timedelta(days=1)

            start_utc = start.astimezone(pytz.UTC)
            end_utc = end.astimezone(pytz.UTC)

            if earliest_start is None or start_utc > earliest_start:
                earliest_start = start_utc
            if latest_end is None or end_utc < latest_end:
                latest_end = end_utc

        return earliest_start, latest_end

    def find_optimal_schedule(self, phone_numbers: List[str], start_time: datetime.time, call_duration: int = 3) -> CallMetrics:
        reference_date = datetime.utcnow().date()
        current_time = datetime.combine(reference_date, start_time).replace(tzinfo=pytz.UTC)
        original_start_time = current_time
        total_call_time = 0
        total_wait_time = 0

        timezone_groups = self.group_by_timezone(phone_numbers)
        remaining_calls = sum(len(calls) for calls in timezone_groups.values())
        logging.debug(f"Total calls to process: {remaining_calls}")

        priority_queue = []
        last_call_end_time = current_time
        day_counter = 0

        # Initial population of priority queue
        for timezones, calls in timezone_groups.items():
            window_start, window_end = self.get_callable_window(timezones, current_time)
            if window_start is not None and window_end is not None:
                time_remaining = int((window_end - max(current_time, window_start)).total_seconds() / 60)
                if time_remaining > 0:
                    heapq.heappush(priority_queue, (time_remaining, timezones, calls))

        while remaining_calls > 0:
            if not priority_queue:
                # Move to next day at 9 AM
                day_counter += 1
                old_time = current_time
                current_time = (current_time.replace(hour=0, minute=0, second=0) + timedelta(days=1)).replace(hour=self.start_hour)

                # Check for remaining calls
                for timezones, calls in timezone_groups.items():
                    unprocessed = [call for call in calls if not call.processed]
                    if unprocessed:
                        window_start, window_end = self.get_callable_window(timezones, current_time)
                        if window_start is not None and window_end is not None:
                            time_remaining = int((window_end - max(current_time, window_start)).total_seconds() / 60)
                            if time_remaining > 0:
                                heapq.heappush(priority_queue, (time_remaining, timezones, unprocessed))

                # Add overnight wait time
                if not priority_queue:
                    overnight_wait = int((current_time - old_time).total_seconds() / 60)
                    total_wait_time += overnight_wait
                    logging.debug(f"Day {day_counter}: Overnight wait: {overnight_wait} minutes")
                continue

            # Process the highest-priority group
            time_remaining, timezones, calls = heapq.heappop(priority_queue)
            unprocessed = [call for call in calls if not call.processed]

            if not unprocessed:
                continue

            window_start, window_end = self.get_callable_window(timezones, current_time)
            if window_start is None or window_end is None:
                continue

            if current_time < window_start:
                wait_time = int((window_start - current_time).total_seconds() / 60)
                if wait_time > 0:
                    total_wait_time += wait_time
                    logging.debug(f"Waiting {wait_time} minutes for window to open")
                    current_time = window_start

            window_duration = int((window_end - current_time).total_seconds() / 60)
            possible_calls = min(len(unprocessed), window_duration // call_duration)

            for i in range(possible_calls):
                if i >= len(unprocessed):
                    break

                call = unprocessed[i]
                call.processed = True
                total_call_time += call_duration
                current_time += timedelta(minutes=call_duration)
                remaining_calls -= 1
                last_call_end_time = current_time

            # Update priority queue
            updated_queue = []
            for _, item_timezones, item_calls in priority_queue:
                unprocessed_item_calls = [call for call in item_calls if not call.processed]
                if unprocessed_item_calls:
                    window_start, window_end = self.get_callable_window(item_timezones, current_time)
                    if window_start is not None and window_end is not None:
                        time_remaining = int((window_end - max(current_time, window_start)).total_seconds() / 60)
                        if time_remaining > 0:
                            heapq.heappush(updated_queue, (time_remaining, item_timezones, unprocessed_item_calls))
            priority_queue = updated_queue

        total_elapsed = int((last_call_end_time - original_start_time).total_seconds() / 60)
        logging.debug(f"Total call time: {total_call_time} minutes")
        logging.debug(f"Total wait time: {total_wait_time} minutes")
        logging.debug(f"Total elapsed time: {total_elapsed} minutes")

        return CallMetrics(agent_call_duration=total_call_time, agent_wait_duration=total_wait_time)

    def _populate_priority_queue(self, priority_queue, timezones, calls, current_time, agent_timezone):
        window_start, window_end = self.get_callable_window(timezones, current_time)
        if window_start is None or window_end is None:
            return  # No valid window, don't add to queue

        agent_local_time = current_time.astimezone(agent_timezone)
        adjusted_window_start = window_start.astimezone(agent_timezone)
        wait_time = max(0, int((adjusted_window_start - agent_local_time).total_seconds() / 60))

        time_remaining = int((window_end - window_start).total_seconds() / 60)
        heapq.heappush(priority_queue, (time_remaining, timezones, calls))


class CallSchedulerApp:
    """Main application class for call scheduling"""

    def __init__(self):
        self.setup_logging()
        self.scheduler = OptimizedCallScheduler()

    def setup_logging(self):
        """Configure logging for the application"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('call_scheduler.log')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def validate_and_parse_time(self, time_str: str) -> Optional[datetime]:
        """
        Validate and convert time string to datetime object
        Expects time in 24-hour format "HH:MM:SS" in UTC
        """
        try:
            # Parse hours, minutes, seconds
            hours, minutes, seconds = map(int, time_str.split(':'))

            # Validate time components
            if not (0 <= hours <= 23 and 0 <= minutes <= 59 and 0 <= seconds <= 59):
                self.logger.error(f"Invalid time values: {time_str}")
                return None

            # Create datetime for today with given time in UTC
            current_date = datetime.now(pytz.UTC).date()
            return datetime.combine(
                current_date,
                time(hours, minutes, seconds),
                tzinfo=pytz.UTC
            )

        except ValueError:
            self.logger.error(f"Invalid time format. Expected HH:MM:SS, got: {time_str}")
            return None

    def validate_input_file(self, file_path: str):
        """
        Validate input CSV file format and content
        Returns True if valid, False otherwise
        """
        valid_phone_numbers = []
        try:
            path = Path(file_path)
            if not path.exists():
                self.logger.error(f"Input file not found: {file_path}")
                return False

            if path.suffix.lower() != '.csv':
                self.logger.error(f"Input file must be CSV format: {file_path}")
                return False

            # Validate file content
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)  # Skip header row

                for i, row in enumerate(reader, 2):  # Start counting from line 2
                    if not row:
                        self.logger.error(f"Empty row found at line {i}")
                        return False
                    if len(row) != 1:
                        self.logger.error(f"Invalid format at line {i}. Expected 1 column, found {len(row)}")
                        return False
                    # Validate phone number format
                    phone_number = row[0]
                    try:
                        parsed_number = phonenumbers.parse(phone_number, "US")  # Assuming US numbers
                        if phonenumbers.is_valid_number(parsed_number):
                            valid_phone_numbers.append(phone_number)  # Add only valid numbers
                        else:
                            print(f"WARNING: Invalid phone number at line {i}, skipping: {phone_number}")

                    except phonenumbers.NumberParseException:
                        print(f"WARNING: Invalid phone number format at line {i}, skipping: {phone_number}")

            return valid_phone_numbers
        except Exception as e:
            self.logger.error(f"Error validating input file: {str(e)}")
            return []

    def validate_start_time(self, start_time_str: str) -> bool:
        """Validate start time format"""
        try:
            # Ensure start_time_str is actually a string before parsing
            if isinstance(start_time_str, datetime):
                start_time_str = start_time_str.strftime("%H:%M:%S")  # Convert datetime to string format

            datetime.strptime(start_time_str, "%H:%M:%S")  # Validate format
            return True
        except ValueError:
            self.logger.error(f"Invalid start time format: {start_time_str}")
            return False

    def read_phone_numbers(self, file_path: str) -> List[str]:
        """Read phone numbers from CSV file, skipping the header"""
        phone_numbers = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip the header row
            phone_numbers = [row[0] for row in reader if row]  # Avoid empty rows
        return phone_numbers

    def format_duration(self, minutes: int) -> str:
        """Format duration in minutes to days, hours, minutes"""
        days = minutes // (24 * 60)
        remaining = minutes % (24 * 60)
        hours = remaining // 60
        mins = remaining % 60

        parts = []
        if days > 0:
            parts.append(f"{days} day{'s' if days != 1 else ''}")
        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if mins > 0:
            parts.append(f"{mins} minute{'s' if mins != 1 else ''}")

        return ", ".join(parts)

    def generate_report(self, metrics: CallMetrics, output_file: str):
        """Generate detailed report of call schedule metrics"""
        with open(output_file, 'w') as f:
            f.write("Call Schedule Analysis Report\n")
            f.write("============================\n\n")

            f.write("Time Breakdown:\n")
            f.write(f"- Total Call Time: {self.format_duration(metrics.agent_call_duration)}\n")
            f.write(f"- Total Wait Time: {self.format_duration(metrics.agent_wait_duration)}\n")
            f.write(f"- Total Duration: {self.format_duration(metrics.total_duration)}\n\n")

            f.write("Efficiency Metrics:\n")
            call_percentage = (metrics.agent_call_duration / metrics.total_duration) * 100
            f.write(f"- Call Time Percentage: {call_percentage:.1f}%\n")
            f.write(f"- Wait Time Percentage: {100 - call_percentage:.1f}%\n")

    def run(self, input_file: str, start_time: str, output_file: str = "report.txt") -> bool:
        """
        Main execution method:
        - Validates and processes the input file.
        - Filters out invalid phone numbers instead of stopping execution.
        - Runs the scheduling algorithm.
        - Writes the output report.
        """
        try:
            # Validate the input file and get valid phone numbers
            phone_numbers = self.validate_input_file(input_file)
            if not phone_numbers:
                self.logger.error("No valid phone numbers found. Exiting...")
                return False  # Stop execution if all numbers are invalid

            # Validate start time format (HH:MM:SS)
            try:
                if isinstance(start_time, datetime):
                    start_time = start_time.strftime("%H:%M:%S")
                start_time_dt = datetime.strptime(start_time, "%H:%M:%S").time()
            except ValueError:
                self.logger.error(f"Invalid start time format: {start_time}")
                return False

            # Process scheduling
            try:
                metrics = self.scheduler.find_optimal_schedule(phone_numbers, start_time_dt)
            except Exception as e:
                self.logger.error(f"Error processing call schedule: {str(e)}")
                return False

            # Write results to output file
            try:
                with open(output_file, "w") as f:
                    f.write("Call Schedule Report\n")
                    f.write("====================\n")
                    f.write(str(metrics))  # Convert metrics to string format
                self.logger.info(f"Report generated successfully: {output_file}")
                return True
            except Exception as e:
                self.logger.error(f"Error writing report: {str(e)}")
                return False

        except Exception as e:
            self.logger.error(f"Unexpected error in run(): {str(e)}")
            return False

            """
            Main execution method
            Returns True if successful, False otherwise
            """
            try:
                # Validate inputs
                if not self.validate_input_file(input_file):
                    return False
                if not self.validate_start_time(start_time):
                    return False

                # Process phone numbers
                self.logger.info(f"Reading phone numbers from {input_file}")
                phone_numbers = self.read_phone_numbers(input_file)
                self.logger.info(f"Found {len(phone_numbers)} phone numbers to process")

                # Calculate schedule
                self.logger.info("Calculating optimal schedule...")
                metrics = self.scheduler.find_optimal_schedule(
                    phone_numbers,
                    datetime.fromisoformat(start_time) if isinstance(start_time, str) else start_time
                )

                # Generate report
                self.logger.info(f"Generating report to {output_file}")
                self.generate_report(metrics, output_file)

                self.logger.info("Processing completed successfully")
                return True

            except Exception as e:
                self.logger.error(f"Error processing call schedule: {str(e)}")
                return False


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Optimize call center schedule")
    parser.add_argument("input_file", help="Path to CSV file containing phone numbers")
    parser.add_argument("--start-time", default="21:30:00",
                        help="Start time in UTC (24-hour format HH:MM:SS)")
    # second file start time is 13:00:00
    parser.add_argument("--output", "-o", default="report.txt",
                        help="Output report file path")

    args = parser.parse_args()

    app = CallSchedulerApp()

    # Parse time at CLI level
    start_time = app.validate_and_parse_time(args.start_time)
    if start_time is None:
        sys.exit(1)

    success = app.run(args.input_file, start_time, args.output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()