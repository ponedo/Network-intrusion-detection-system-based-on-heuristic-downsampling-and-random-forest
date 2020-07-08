attack_types = {
    "0_Normal": ["normal"], #为处理方便将正常类型一同加入攻击类型中

    "1_DOS": ["back","land","neptune","pod","smurf","teardrop","mailbomb","processtable","udpstorm","apache2",],

    "2_Probe": ["satan","ipsweep","nmap","portsweep","mscan","saint",],

    "3_R2L": ["guess_passwd","ftp_write","imap","phf","multihop","warezmaster","xlock","xsnoop","snmpguess",
            "snmpgetattack","sendmail","named","worm","warezclient","spy"],

    "4_U2R": ["buffer_overflow","loadmodule","rootkit","perl","sqlattack","xterm","ps","httptunnel",]
}

feature_list = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 
    'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 
    'attack_type', 'difficulty'
]

numeric_int_feature_list = [
    'duration', 'src_bytes', 'dst_bytes', 'num_failed_logins', 'hot', 'num_compromised', 
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 
    'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count'
]

numeric_float_feature_list = [
    'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

categorical_multi_feature_list = [
    'protocol_type', 'flag', 'service',
]

categorical_binary_feature_list = [
    'land', 'su_attempted', 'logged_in', 'is_host_login', 'root_shell', 'is_guest_login'
]

categorical_multi_feature_metalist = {
    'protocol_type': {'tcp', 'icmp', 'udp'}, 
    'flag': {'S2', 'REJ', 'RSTOS0', 'S0', 'SF', 'SH', 'S1', 'RSTO', 'S3', 'RSTR', 'OTH'}, 
    'service': {
        'domain', 'eco_i', 'tftp_u', 'systat', 'Z39_50', 'courier', 'bgp', 'http', 'http_443', 
        'http_8001', 'http_2784', 'sql_net', 'netbios_ns', 'mtp', 'klogin', 'kshell', 'printer', 
        'efs', 'X11', 'tim_i', 'pop_2', 'urh_i', 'link', 'uucp_path', 'gopher', 'vmnet', 'whois', 
        'sunrpc', 'pop_3', 'csnet_ns', 'supdup', 'domain_u', 'ctf', 'daytime', 'nnsp', 'shell', 
        'imap4', 'netstat', 'exec', 'urp_i', 'finger', 'ssh', 'discard', 'telnet', 'other', 'hostnames', 
        'netbios_dgm', 'aol', 'IRC', 'remote_job', 'ntp_u', 'harvest', 'iso_tsap', 'echo', 'private', 
        'ftp_data', 'netbios_ssn', 'ldap', 'rje', 'ftp', 'time', 'ecr_i', 'auth', 'name', 'login', 
        'pm_dump', 'red_i', 'nntp', 'smtp', 'uucp'
    }
}