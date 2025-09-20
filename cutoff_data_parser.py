import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple, Optional

class CutoffDataParser:
    def __init__(self):
        self.cutoff_data = {}
        self.college_branch_combinations = set()
        
    def parse_rank_percentile(self, rank_percentile_str: str) -> Tuple[Optional[int], Optional[float]]:
        """
        Parse rank and percentile from string like '14789 (85.2493707)'
        Returns (rank, percentile) or (None, None) if parsing fails
        """
        if pd.isna(rank_percentile_str) or rank_percentile_str == '':
            return None, None
            
        try:
            # Extract rank and percentile using regex
            match = re.search(r'(\d+)\s*\(([\d.]+)\)', str(rank_percentile_str))
            if match:
                rank = int(match.group(1))
                percentile = float(match.group(2))
                return rank, percentile
            else:
                # Try to extract just the number if no parentheses
                numbers = re.findall(r'[\d.]+', str(rank_percentile_str))
                if len(numbers) >= 1:
                    if len(numbers) == 1:
                        # If only one number, assume it's percentile if < 100, rank otherwise
                        num = float(numbers[0])
                        if num <= 100:
                            return None, num
                        else:
                            return int(num), None
                    else:
                        # Multiple numbers, take first as rank, second as percentile
                        return int(float(numbers[0])), float(numbers[1])
        except (ValueError, AttributeError):
            pass
            
        return None, None
    
    def clean_college_name(self, college_name: str) -> str:
        """Clean and standardize college names"""
        if pd.isna(college_name) or college_name == '':
            return ''
            
        # Remove choice codes and extra formatting
        college_name = str(college_name).strip()
        
        # Remove patterns like "01101 - " from the beginning
        college_name = re.sub(r'^\d+\s*-\s*', '', college_name)
        
        # Clean up common formatting issues
        college_name = re.sub(r'\s+', ' ', college_name)  # Multiple spaces to single
        college_name = college_name.strip()
        
        return college_name
    
    def clean_branch_name(self, branch_name: str) -> str:
        """Clean and standardize branch names"""
        if pd.isna(branch_name) or branch_name == '':
            return ''
            
        branch_name = str(branch_name).strip()
        
        # Standardize common branch names
        branch_mappings = {
            'Computer Science and Engineering': 'Computer Science Engineering',
            'Electronics and Telecommunication Engg': 'Electronics and Telecommunication Engineering',
            'Electrical Engg[Electronics and Power]': 'Electrical Engineering',
            'Artificial Intelligence (AI) and Data Science': 'Artificial Intelligence and Data Science',
            'Computer Science and Engineering (IoT)': 'Computer Science Engineering (IoT)',
            'Computer Science and Engineering(Artificial': 'Computer Science Engineering (AI)',
            'Artificial Intelligence and Data Science': 'Artificial Intelligence and Data Science'
        }
        
        for old_name, new_name in branch_mappings.items():
            if old_name in branch_name:
                branch_name = new_name
                break
                
        return branch_name.strip()
    
    def load_cutoff_data(self, file_path: str, year: int) -> pd.DataFrame:
        """Load and parse cutoff data from CSV file"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"Successfully loaded {file_path} with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError(f"Could not read {file_path} with any of the tried encodings")
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Parse the data
            parsed_data = []
            
            for _, row in df.iterrows():
                # Skip empty rows
                if pd.isna(row.iloc[0]) or row.iloc[0] == '':
                    continue
                    
                # Extract rank and percentile
                rank, percentile = self.parse_rank_percentile(row.iloc[1])
                
                if rank is None and percentile is None:
                    continue
                
                # Clean college and branch names
                college_name = self.clean_college_name(row.iloc[3])
                branch_name = self.clean_branch_name(row.iloc[4])
                
                if not college_name or not branch_name:
                    continue
                
                # Extract exam type
                exam_type = str(row.iloc[5]).strip() if not pd.isna(row.iloc[5]) else 'Unknown'
                
                parsed_data.append({
                    'year': year,
                    'rank': rank,
                    'percentile': percentile,
                    'college': college_name,
                    'branch': branch_name,
                    'exam_type': exam_type,
                    'choice_code': str(row.iloc[2]).strip() if not pd.isna(row.iloc[2]) else '',
                    'seat_type': str(row.iloc[7]).strip() if len(row) > 7 and not pd.isna(row.iloc[7]) else 'AI'
                })
            
            return pd.DataFrame(parsed_data)
            
        except Exception as e:
            print(f"Error loading cutoff data from {file_path}: {e}")
            return pd.DataFrame()
    
    def load_all_cutoff_data(self) -> pd.DataFrame:
        """Load cutoff data from both 2024 and 2025 files"""
        all_data = []
        
        # Load 2024 data
        print("Loading 2024 cutoff data...")
        df_2024 = self.load_cutoff_data('cuttoff2024.csv', 2024)
        if not df_2024.empty:
            all_data.append(df_2024)
            print(f"Loaded {len(df_2024)} records from 2024")
        
        # Load 2025 data
        print("Loading 2025 cutoff data...")
        df_2025 = self.load_cutoff_data('cuttoff2025.csv', 2025)
        if not df_2025.empty:
            all_data.append(df_2025)
            print(f"Loaded {len(df_2025)} records from 2025")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"Total records loaded: {len(combined_df)}")
            
            # Create college-branch combinations set
            self.college_branch_combinations = set(
                zip(combined_df['college'], combined_df['branch'])
            )
            
            return combined_df
        else:
            print("No data loaded!")
            return pd.DataFrame()
    
    def get_college_branch_stats(self, df: pd.DataFrame) -> Dict:
        """Get statistics about colleges and branches"""
        stats = {
            'total_records': len(df),
            'unique_colleges': df['college'].nunique(),
            'unique_branches': df['branch'].nunique(),
            'unique_combinations': len(self.college_branch_combinations),
            'years_covered': sorted(df['year'].unique().tolist()),
            'exam_types': df['exam_type'].value_counts().to_dict(),
            'top_colleges': df['college'].value_counts().head(10).to_dict(),
            'top_branches': df['branch'].value_counts().head(10).to_dict()
        }
        return stats
    
    def get_cutoff_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze cutoff trends for college-branch combinations"""
        # Group by college and branch, then analyze trends
        trends = df.groupby(['college', 'branch']).agg({
            'percentile': ['min', 'max', 'mean', 'count'],
            'rank': ['min', 'max', 'mean'],
            'year': ['min', 'max']
        }).round(2)
        
        # Flatten column names
        trends.columns = ['_'.join(col).strip() for col in trends.columns]
        trends = trends.reset_index()
        
        # Calculate trend direction (increasing/decreasing cutoff)
        trends['cutoff_trend'] = trends.apply(
            lambda row: 'Increasing' if row['percentile_min'] < row['percentile_max'] else 
                       'Decreasing' if row['percentile_min'] > row['percentile_max'] else 'Stable',
            axis=1
        )
        
        return trends.sort_values('percentile_mean', ascending=False)
    
    def find_eligible_colleges(self, student_percentile: float, student_rank: Optional[int], 
                             branch: str, year: int = 2025) -> pd.DataFrame:
        """Find colleges where student is eligible based on cutoff data"""
        if df.empty:
            return pd.DataFrame()
            
        # Filter by branch and year
        filtered_df = df[(df['branch'].str.contains(branch, case=False, na=False)) & 
                        (df['year'] == year)].copy()
        
        if filtered_df.empty:
            return pd.DataFrame()
        
        # Filter by percentile (student should have higher percentile than cutoff)
        if student_percentile is not None:
            eligible_by_percentile = filtered_df[filtered_df['percentile'] <= student_percentile]
        else:
            eligible_by_percentile = filtered_df
        
        # Filter by rank (student should have better rank than cutoff)
        if student_rank is not None:
            eligible_by_rank = filtered_df[filtered_df['rank'] >= student_rank]
        else:
            eligible_by_rank = filtered_df
        
        # Combine both criteria
        eligible = pd.concat([eligible_by_percentile, eligible_by_rank]).drop_duplicates()
        
        # Sort by percentile (ascending - easier to get into)
        eligible = eligible.sort_values('percentile', ascending=True)
        
        return eligible

# Global variable to store the loaded data
df = pd.DataFrame()

def initialize_cutoff_data():
    """Initialize the cutoff data"""
    global df
    parser = CutoffDataParser()
    df = parser.load_all_cutoff_data()
    return parser, df

if __name__ == "__main__":
    # Test the parser
    parser, data = initialize_cutoff_data()
    
    if not data.empty:
        print("\n" + "="*50)
        print("CUTOFF DATA ANALYSIS")
        print("="*50)
        
        # Get statistics
        stats = parser.get_college_branch_stats(data)
        print(f"\nData Statistics:")
        print(f"- Total records: {stats['total_records']}")
        print(f"- Unique colleges: {stats['unique_colleges']}")
        print(f"- Unique branches: {stats['unique_branches']}")
        print(f"- Years covered: {stats['years_covered']}")
        
        print(f"\nTop 5 Colleges by number of programs:")
        for college, count in list(stats['top_colleges'].items())[:5]:
            print(f"  - {college}: {count} programs")
        
        print(f"\nTop 5 Branches:")
        for branch, count in list(stats['top_branches'].items())[:5]:
            print(f"  - {branch}: {count} programs")
        
        # Test eligibility check
        print(f"\n" + "="*30)
        print("TESTING ELIGIBILITY CHECK")
        print("="*30)
        
        test_cases = [
            (95.0, 10000, "Computer Science"),
            (85.0, 25000, "Mechanical"),
            (90.0, 15000, "Electronics")
        ]
        
        for percentile, rank, branch in test_cases:
            print(f"\nStudent: {percentile}% percentile, Rank {rank}, Branch: {branch}")
            eligible = parser.find_eligible_colleges(percentile, rank, branch)
            if not eligible.empty:
                print(f"  Found {len(eligible)} eligible options:")
                for _, row in eligible.head(3).iterrows():
                    print(f"    - {row['college']}: {row['percentile']:.2f}% (Rank: {row['rank']})")
            else:
                print("  No eligible colleges found")
    else:
        print("Failed to load cutoff data!")
