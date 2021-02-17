#! /usr/bin/env python3
import paramiko
import argparse
import time
from scp import SCPClient

class SSHConnection(object):
    def __init__(self, host, username, password, port=22):
        self.host = host
        self.password = password
        self.username = username
        self.window_size = pow(4, 12)#about ~16MB chunks
        self.max_packet_size = pow(4, 12)
        self.sftp = None
        self.nuc = None
        self.evm = None
    
    #Create a Paramiko SSHClient object which can be used to open a SFTP connection.
    def createSSHClient(self):
        self.nuc = paramiko.SSHClient()
        self.nuc.load_system_host_keys()
        self.nuc.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.nuc.connect(self.host, username=self.username,
                         password=self.password, allow_agent=False,look_for_keys=False)
        nuctransport = self.nuc.get_transport()
        dest_addr = ('192.168.1.1', 22)
        local_addr = (self.host, 22)
        vmchannel = nuctransport.open_channel(
            "direct-tcpip", dest_addr, local_addr, self.window_size, self.max_packet_size)

        self.evm = paramiko.SSHClient()
        self.evm.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.evm.connect('192.168.1.1', username='root',
                         password='', sock=vmchannel, allow_agent=False,look_for_keys=False)

    def executeRemoteInference(self, remote_path, local_path):
        
        stdin, stdout, stderr = self.evm.exec_command('new_SimpleNN ' + remote_path+'/test.blob ' + remote_path+'/input-0.bin ' + remote_path+'/output.dat ' + remote_path+'/expected.dat ' + '0')
        lines = stdout.readlines()
        for line in lines:
            print(line)
            if(line == 'Applying Softmax to results to create probability distribution...\n'):
                print('INFERENCE SUCCESS! The output is located at: ' + local_path )
         
    def get(self, remote_path, local_path):
        self.sftp.get(remote_path, local_path)

    def put(self, local_path, remote_path, remote_file):
        self.sftp = self.evm.open_sftp()
        try:
            self.sftp.chdir(remote_path)  # Test if remote_path exists
        except IOError:
            self.sftp.mkdir(remote_path)  # Create remote_path
        self.sftp.put(local_path, remote_file)
        

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Runs an inference using SimpleNN on a remote EVM.')
    parser.add_argument('--user', type=str, required=True, help='Your name so that a folder can be created on the EVM for your blobs/inputs')
    parser.add_argument('--evmIP', type=str, required=True, help='EVM IP address')
    parser.add_argument('--evmUserName', type=str, required=True, help='EVM login username')
    parser.add_argument('--evmPassword', type=str, required=True, help='EVM login password')
    parser.add_argument('--blob', type=str, required=True, help='path to test.blob')
    parser.add_argument('--input', type=str, required=True, help='path to input-0.bin')
    parser.add_argument('--output', type=str, required=True, help='path to where local output from inference will be stored')

    args = parser.parse_args()

    ssh = SSHConnection(args.evmIP, args.evmUserName, args.evmPassword)
    ssh.createSSHClient()

    remote_path = '/home/root/'+args.user+'testFolder'
    #SFTP input-0.bin and test.blob to EVM
    ssh.put(args.blob, remote_path,remote_path+'/test.blob')
    ssh.put(args.input, remote_path,remote_path+'/input-0.bin')

    #Do Inference with SimpleNN
    ssh.executeRemoteInference(remote_path, args.output)

    #SCP output from inference back to host
    ssh.get(remote_path+'/output.dat', args.output)
    print("Total inference Took %d seconds " % (time.time() - start_time))
   
if __name__ == "__main__":
    main()


