using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net.Sockets;
using System.Net;
using System;
using System.IO;

public class EnvironmentCommunicator : MonoBehaviour {

	public int port = 2851;
	public const int maxImageSize = 256;

	private CameraSensor cameraSensor;
	private CarController carController;
	private CarState carState;

	private Socket listener;
	private Client client;

	

	private class Checkpoints {  
		public GameObject[] checkpointsArray;
		private int currentCheckpointNumber;
		private GameObject backCheckpoint;

		public Checkpoints() {
			this.checkpointsArray = GameObject.FindGameObjectsWithTag("Checkpoint");
			//Array.Reverse(this.checkpointsArray);

					System.Array.Sort(this.checkpointsArray,
    		delegate(GameObject x, GameObject y) { return x.name.CompareTo(y.name); });

			this.backCheckpoint = this.checkpointsArray[this.checkpointsArray.Length - 1];

		// 			foreach(GameObject ch in this.checkpointsArray){
		// 	Debug.Log("Checkpoint init " + ch.name);
		// }

			this.currentCheckpointNumber = 0;
		}

		public void NextCheckpoint() {
			this.backCheckpoint = this.getCurrentCheckpoint();
			this.currentCheckpointNumber = (this.currentCheckpointNumber + 1) % this.checkpointsArray.Length;
		}

		public GameObject getCurrentCheckpoint() {
			return this.checkpointsArray[this.currentCheckpointNumber];
		}

		public GameObject[] getCheckpoints() {
			return this.checkpointsArray;
		}

		public Vector3 getLineBetweenAdjacentheckpoints(){
			return this.getCurrentCheckpoint().transform.position - this.backCheckpoint.transform.position;

		}

		public int getCurrentCheckpointNumber() {
			return this.currentCheckpointNumber;
		}
	}

	private class Reward {
		private Checkpoints checkpoints;

		private float lastReward;
		private float lastCheckpointEpisodeReward;

		private GameObject forwardMark;
		private GameObject backwardMark;

		public Reward(){
			this.lastReward = 0;
			this.lastCheckpointEpisodeReward = 0;

			this.forwardMark = GameObject.FindGameObjectsWithTag("ForwardCarMark")[0];
			this.backwardMark = GameObject.FindGameObjectsWithTag("BackwardCarMark")[0];

			//Debug.Log(this.forwardMark, this.backwardMark);
		}

		public Reward(Checkpoints checkpoints) {
			this.checkpoints = checkpoints;

						this.forwardMark = GameObject.FindGameObjectsWithTag("ForwardCarMark")[0];
			this.backwardMark = GameObject.FindGameObjectsWithTag("BackwardCarMark")[0];
		}

		public float getReward(Transform currentTransform) {

			float rewardNow = /*this.lastCheckpointEpisodeReward +*/ 10/(this.checkpoints.getCurrentCheckpoint().transform.position - currentTransform.position).magnitude;

			Vector3 mainDirectionLine = this.checkpoints.getLineBetweenAdjacentheckpoints();
			mainDirectionLine.y = 0;

			Vector3 forwardPoint = this.checkpoints.getCurrentCheckpoint().transform.position - this.forwardMark.transform.position;
			forwardPoint.y = 0;

			Vector3 backwardPoint = this.checkpoints.getCurrentCheckpoint().transform.position - this.backwardMark.transform.position;
			backwardPoint.y = 0;

			float angleForward = Vector3.Angle(mainDirectionLine, forwardPoint);
			float angleBackward = Vector3.Angle(mainDirectionLine, backwardPoint);
			//Debug.Log("forward " + this.forwardMark.transform.position);
			//Debug.Log(currentDirection);
			// Debug.Log("forward " +  /*Mathf.Rad2Deg * */);
			// Debug.Log("backward " +  /*Mathf.Rad2Deg **/ );


			float sign = 1;

			if (angleForward - angleBackward > 5) {
				sign = -1;
			} else {
				sign = 1;
			}
			float reward = 500 * (rewardNow - this.lastReward) * sign;

			

			if (reward > 5) {
				reward = 5;
			} else if (reward < 0) {
				reward = 0;
			}

			//float dev = (reward * 0xffffff) / 0xffffff;
			//Debug.Log("my reward " + reward + " / " +  ((int)(reward * 0xffffff)) + " / " + dev);

			this.lastReward = rewardNow;

			return reward ;
		}

		public void appendPrevCheckpointReward(){
			this.lastCheckpointEpisodeReward = lastReward;
		}


	}

	void OnTriggerEnter(Collider other) {
		if (other.tag == "Checkpoint" && other.name == checkpoints.getCurrentCheckpoint().name) {
			//other.gameObject.SetActive(false);

			this.reward.appendPrevCheckpointReward();
			//Debug.Log("Last checkpoint # " + this.checkpoints.getCurrentCheckpointNumber());

			this.checkpoints.NextCheckpoint();
			//Debug.Log("Next checkpoint # " + this.checkpoints.getCurrentCheckpointNumber());
			Debug.Log("Car passed checkpoint");
		
		}
	}

	private Checkpoints checkpoints;
	private Reward reward;

	private const byte REQUEST_READ_SENSORS = 1;
	private const byte REQUEST_WRITE_ACTION = 2;

	private class Client {

		public Client(Socket socket) {
			this.socket = socket;
		}

		public void BeginReceive(AsyncCallback callback) {
			socket.BeginReceive(buffer, 0, BufferSize, 0, callback, this);
		}

		public Socket socket;
		public const int BufferSize = 1024;
		public int bytesLeft = 0;
		public byte[] buffer = new byte[BufferSize];
		public volatile bool requestPending;
		public const int ResponseBufferSize = maxImageSize * maxImageSize + 16;
		public byte[] responseBuffer = new byte[ResponseBufferSize];
	}

	private int DeterminePort() {
		string[] args = System.Environment.GetCommandLineArgs();
		for(int i = 0; i < args.Length; i++) {
			if(args[i] == "--port" && i + 1 < args.Length)
				return Convert.ToInt32(args[i + 1]);
		}
		// or return default port
		return port;
	}

	void Start() {
		cameraSensor = GetComponentInChildren<CameraSensor>();
		carState = GetComponent<CarState>();
		carController = GetComponent<CarController>();

		IPEndPoint localEndPoint = new IPEndPoint(IPAddress.Any, DeterminePort());

		listener = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);

		checkpoints	= new Checkpoints();
		reward = new Reward(checkpoints);


		

		// foreach (GameObject gameObject in checkpoints.checkpointsArray) {
		// 	Debug.Log(gameObject.name);
		// }

		try {
			listener.Bind(localEndPoint);
			listener.Listen(1);

			BeginAccept();
		} catch (Exception e) {
			Debug.LogException(e);
		}
	}

	private void BeginAccept() {
		listener.BeginAccept(new AsyncCallback(AcceptCallback), listener);
	}

	private void AcceptCallback(IAsyncResult ar) {
		Socket listener = (Socket) ar.AsyncState;
		Socket handler = listener.EndAccept(ar);

		Debug.Log("Accepted new client");

		client = new Client(handler);
		client.BeginReceive(new AsyncCallback(ReadCallback));
	}

	private void ReadCallback(IAsyncResult ar) {  
		int bytesRead = client.socket.EndReceive(ar);  

		if (bytesRead > 0) {
			client.bytesLeft = bytesRead;
			client.requestPending = true;
		}
	}

	private void SendSensorData(int imageWidth, int imageHeight) {
		imageWidth = Math.Min(imageWidth, maxImageSize);
		imageHeight = Math.Min(imageHeight, maxImageSize);
		cameraSensor.TakePicture(image => {
			int responseSize = 6 + imageWidth * imageHeight;
			BinaryWriter writer = new BinaryWriter(new MemoryStream(client.responseBuffer));
			writer.Write(carState.Disqualified);
			writer.Write(carState.Finished);
			//Debug.Log("velo" + (carController.GetVelocity() * 0xffff));
			// We can't transfer floating point values, so let's transmit velocity * 2^16
			int velocity = IPAddress.HostToNetworkOrder((int)(reward.getReward(this.carState.transform) * 0xffffff)/*carController.GetVelocity() * 0xffff*/ );
			writer.Write(velocity);
			carState.ResetState();
			WriteCameraImage(writer, image);
			client.socket.BeginSend(client.responseBuffer, 0, responseSize, 0, new AsyncCallback(SendCallback), client);
		}, imageWidth, imageHeight);
	}

	private void ApplyAction(int vertical, int horizontal) {
		carController.ApplyAction(vertical, horizontal);
	}

	void Update() {
		//Debug.Log(reward.getReward(this.carState.transform));
		if(client != null && client.requestPending) {
			client.requestPending = false;
			BinaryReader reader = new BinaryReader(new MemoryStream(client.buffer));
			while(client.bytesLeft > 0) {
				byte instruction = reader.ReadByte();
				client.bytesLeft--;
				switch(instruction) {
				case REQUEST_READ_SENSORS:
					int width = IPAddress.NetworkToHostOrder(reader.ReadInt32());
					int height = IPAddress.NetworkToHostOrder(reader.ReadInt32());
					client.bytesLeft -= 8;
					SendSensorData(width, height);
					break;
				case REQUEST_WRITE_ACTION:
					int vertical = IPAddress.NetworkToHostOrder(reader.ReadInt32());
					int horizontal = IPAddress.NetworkToHostOrder(reader.ReadInt32());
					client.bytesLeft -= 8;
					ApplyAction(vertical, horizontal);
					break;
				}
			}
			// Listen for next message
			client.BeginReceive(new AsyncCallback(ReadCallback));
		}
	}

	private void SendCallback(IAsyncResult ar) {
		client.socket.EndSend(ar);
	}  

	private void WriteCameraImage(BinaryWriter writer, Texture2D image) {
		for(int y = 0; y < image.height; y++) {
			for(int x = 0; x < image.width; x++) {
				Color pixel = image.GetPixel(x, y);
				 byte grayscale = 0;
				// if ( y < image.height * 0.5){
					grayscale = (byte)(pixel.grayscale * 255);
				// }
				writer.Write(grayscale);
			}
		}
	}
}