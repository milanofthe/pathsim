import requests
import os
from datetime import datetime

def fetch_github_issues(app, config):
    """Fetch GitHub issues and generate RST file"""
    
    # Configure these
    repo = "milanofthe/pathsim" 
    token = os.environ.get('GITHUB_TOKEN')
    
    # API request
    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {'Authorization': f'token {token}'} if token else {}
    params = {
        'state': 'all',
        'sort': 'created',
        'direction': 'desc',
        'per_page': 100
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        issues = response.json()
        
        # Filter out pull requests
        issues = [i for i in issues if 'pull_request' not in i]
        
        # Generate RST file
        output_path = os.path.join(app.srcdir, 'roadmap_generated.rst')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(".. raw:: html\n\n")
            f.write("   <div class='github-issues-container'>\n")
            f.write(f"   <p class='issues-updated'>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</p>\n\n")
            
            # Group by state
            open_issues = [i for i in issues if i['state'] == 'open']
            
            if open_issues:
                for issue in open_issues:
                    # Write only 'roadmap' labeled issues
                    if any(label['name'] == 'roadmap' for label in issue.get('labels', [])):
                        write_issue_html(f, issue)
            else:
                f.write("   <p class='no-issues'>No roadmap items found.</p>\n")
            
            f.write("   </div>\n\n")
        
        print(f"Generated roadmap from {len(issues)} GitHub issues")
        
    except Exception as e:
        print(f"Warning: Could not fetch GitHub issues: {e}")
        output_path = os.path.join(app.srcdir, 'roadmap_generated.rst')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(".. raw:: html\n\n")
            f.write("   <div class='github-issues-container'>\n")
            f.write("   <p class='issues-error'>⚠️ Could not fetch issues from GitHub</p>\n")
            f.write("   </div>\n\n")

def write_issue_html(f, issue):
    """Write a single issue in HTML format for better styling"""
    labels = issue.get('labels', [])
    body = issue.get('body', '').strip()
    
    # Truncate body
    if len(body) > 400:
        body = body[:400] + "..."
    
    # Escape HTML characters
    body = body.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    body = body.replace('\n', '<br>')
    
    f.write("   <div class='github-issue-card'>\n")
    f.write("      <div class='issue-header'>\n")
    f.write(f"         <span class='issue-number'>#{issue['number']}</span>\n")
    f.write(f"         <h3 class='issue-title'>{issue['title']}</h3>\n")
    f.write("      </div>\n")
    
    if labels:
        f.write("      <div class='issue-labels'>\n")
        for label in labels:
            f.write(f"         <span class='issue-label'>{label['name']}</span>\n")
        f.write("      </div>\n")
    
    if body:
        f.write("      <div class='issue-body'>\n")
        f.write(f"         <p>{body}</p>\n")
        f.write("      </div>\n")
    
    f.write("      <div class='issue-footer'>\n")
    created_date = datetime.strptime(issue['created_at'], '%Y-%m-%dT%H:%M:%SZ').strftime('%b %d, %Y')
    f.write(f"         <span class='issue-date'>Created: {created_date}</span>\n")
    f.write(f"         <a href='{issue['html_url']}' class='issue-link' target='_blank'>View on GitHub →</a>\n")
    f.write("      </div>\n")
    f.write("   </div>\n\n")

def setup(app):
    """Sphinx extension setup"""
    app.connect('config-inited', fetch_github_issues)
    
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }