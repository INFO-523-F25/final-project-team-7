# NSL-KDD Dataset

## Data
- **Dataset:** The NSL-KDD dataset is an improved version of the original KDD Cup 1999 dataset, designed for evaluating intrusion detection systems (IDS).  
  It contains a variety of simulated network traffic records labeled as either *normal* or one of several *attack types*.  
  Each record represents a single network connection and includes 41 features extracted from packet and connection metadata.  
  The dataset addresses issues in the original KDD Cup 99 dataset such as redundant records and unbalanced attack distributions, making it more suitable for training and testing IDS models.

---

## Codebook for NSL-KDD Dataset

### Variable Names and Descriptions

| Variable | Description |
|-----------|-------------|
| `duration` | Length (in seconds) of the network connection |
| `protocol_type` | Type of protocol (e.g., tcp, udp, icmp) |
| `service` | Network service on the destination (e.g., http, ftp, telnet) |
| `flag` | Status flag of the connection (e.g., SF, REJ) |
| `src_bytes` | Number of data bytes sent from source to destination |
| `dst_bytes` | Number of data bytes sent from destination to source |
| `land` | 1 if connection is from/to the same host/port; 0 otherwise |
| `wrong_fragment` | Number of wrong fragments |
| `urgent` | Number of urgent packets |
| `hot` | Number of "hot" indicators (e.g., access to system directories) |
| `num_failed_logins` | Number of failed login attempts |
| `logged_in` | 1 if successfully logged in; 0 otherwise |
| `num_compromised` | Number of compromised conditions |
| `root_shell` | 1 if root shell obtained; 0 otherwise |
| `su_attempted` | 1 if "su root" command attempted; 0 otherwise |
| `num_root` | Number of root accesses |
| `num_file_creations` | Number of file creation operations |
| `num_shells` | Number of shell prompts invoked |
| `num_access_files` | Number of operations on access control files |
| `num_outbound_cmds` | Number of outbound commands (always 0 in this dataset) |
| `is_host_login` | 1 if connection is a host login; 0 otherwise |
| `is_guest_login` | 1 if connection is a guest login; 0 otherwise |
| `count` | Number of connections to the same host in the past 2 seconds |
| `srv_count` | Number of connections to the same service in the past 2 seconds |
| `serror_rate` | % of connections that have SYN errors |
| `srv_serror_rate` | % of same-service connections that have SYN errors |
| `rerror_rate` | % of connections that have REJ errors |
| `srv_rerror_rate` | % of same-service connections that have REJ errors |
| `same_srv_rate` | % of connections to the same service |
| `diff_srv_rate` | % of connections to different services |
| `srv_diff_host_rate` | % of connections to different hosts |
| `dst_host_count` | Number of connections to the same destination host |
| `dst_host_srv_count` | Number of connections to the same service as the destination host |
| `dst_host_same_srv_rate` | % of destination host connections to same service |
| `dst_host_diff_srv_rate` | % of destination host connections to different services |
| `dst_host_same_src_port_rate` | % of destination host connections from same source port |
| `dst_host_srv_diff_host_rate` | % of destination host connections to different hosts |
| `dst_host_serror_rate` | % of destination host connections with SYN errors |
| `dst_host_srv_serror_rate` | % of destination host same-service connections with SYN errors |
| `dst_host_rerror_rate` | % of destination host connections with REJ errors |
| `dst_host_srv_rerror_rate` | % of destination host same-service connections with REJ errors |
| `label` | Type of network connection (normal or attack category) |
| `attack_type` | Attack category (DoS, Probe, R2L, U2R) |

---

### Data Types

| Column | Data Type |
|---------|------------|
| `duration` | int64 |
| `protocol_type` | object |
| `service` | object |
| `flag` | object |
| `src_bytes` | int64 |
| `dst_bytes` | int64 |
| `land` | int64 |
| `wrong_fragment` | int64 |
| `urgent` | int64 |
| `hot` | int64 |
| `num_failed_logins` | int64 |
| `logged_in` | int64 |
| `num_compromised` | int64 |
| `root_shell` | int64 |
| `su_attempted` | int64 |
| `num_root` | int64 |
| `num_file_creations` | int64 |
| `num_shells` | int64 |
| `num_access_files` | int64 |
| `num_outbound_cmds` | int64 |
| `is_host_login` | int64 |
| `is_guest_login` | int64 |
| `count` | int64 |
| `srv_count` | int64 |
| `serror_rate` | float64 |
| `srv_serror_rate` | float64 |
| `rerror_rate` | float64 |
| `srv_rerror_rate` | float64 |
| `same_srv_rate` | float64 |
| `diff_srv_rate` | float64 |
| `srv_diff_host_rate` | float64 |
| `dst_host_count` | int64 |
| `dst_host_srv_count` | int64 |
| `dst_host_same_srv_rate` | float64 |
| `dst_host_diff_srv_rate` | float64 |
| `dst_host_same_src_port_rate` | float64 |
| `dst_host_srv_diff_host_rate` | float64 |
| `dst_host_serror_rate` | float64 |
| `dst_host_srv_serror_rate` | float64 |
| `dst_host_rerror_rate` | float64 |
| `dst_host_srv_rerror_rate` | float64 |
| `label` | object |
| `attack_type` | object |



