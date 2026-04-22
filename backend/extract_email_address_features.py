import re
from collections import Counter
import numpy as np

def extract_email_address_features(email_addr):
    """
    Extract 8 phishing-related features from sender email address.
    """
    if not email_addr or '@' not in email_addr:
        return [0.0] * 8
    
    try:
        local, domain = email_addr.split('@', 1)
        domain = domain.lower().strip()
        
        # 1. email_addr_len
        email_addr_len = len(email_addr)
        
        # 2. domain_len
        domain_len = len(domain)
        
        # 3. suspicious_tld (1 if risky TLDs like .tk, .ml)
        suspicious_tlds = {'tk', 'ml', 'ga', 'cf', 'gq', 'top', 'ru', 'cn', 'xyz', 'club'}
        tld = domain.split('.')[-1]
        suspicious_tld = 1.0 if tld in suspicious_tlds else 0.0
        
        # 4. free_domain (1 if common free like gmail)
        free_domains = {'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'mail.com'}
        free_domain = 1.0 if any(fd in domain for fd in free_domains) else 0.0
        
        # 5. num_dots_domain
        num_dots_domain = float(domain.count('.'))
        
        # 6. has_digits_addr (1 if digits in full addr)
        has_digits_addr = 1.0 if any(c.isdigit() for c in email_addr) else 0.0
        
        # 7. domain_entropy (char diversity)
        char_counts = Counter(domain)
        total_chars = len(domain)
        domain_entropy = -sum((count / total_chars) * np.log2(count / total_chars + 1e-10) for count in char_counts.values()) if total_chars > 0 else 0.0
        
        # 8. has_plus (1 if +alias)
        has_plus = 1.0 if '+' in local else 0.0
        
        return [float(email_addr_len), float(domain_len), suspicious_tld, free_domain, num_dots_domain, has_digits_addr, float(domain_entropy), has_plus]
    
    except Exception:
        return [0.0] * 8

