using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Text;
using System.Globalization; 
using System;

namespace EVP
{
public class CameraRender : MonoBehaviour
{
    public int textureWidth = 1024;
    public int textureHeight = 1024;
    public Camera renderCamera;
    public Image uiImage; // Referência para a UI Image
    public RawImage rawImage; // Alternativa usando RawImage
    private RenderTexture renderTexture;

    private const int PORT_SEND = 5000;    // Porta para enviar imagens
    private const int PORT_RECEIVE = 5001; // Porta para receber comandos
    private string serverIP = "127.0.0.1";
    
    private UdpClient sendClient;
    private UdpClient receiveClient;
    private Thread receiveThread;
    private bool isRunning = true;
    

    public float speed=0;
    public float steering=0;


    public VehicleController target;

    private Vector3 initialPosition;
    private Quaternion initialRotation;
    private Vector3 initialVelocity;
    private Vector3 initialAngularVelocity;



    
    void Start()
    {

     if (target != null)
        {
            initialPosition = target.transform.position;
            initialRotation = target.transform.rotation;
            initialVelocity = Vector3.zero;
            initialAngularVelocity = Vector3.zero;
            

            if (target.GetComponent<Rigidbody>() != null)
            {
                Rigidbody rb = target.GetComponent<Rigidbody>();
                initialVelocity = rb.velocity;
                initialAngularVelocity = rb.angularVelocity;
            }
        }

     
        SetupCamera();
        SetupNetwork();
        StartReceiving();
    }

     void OnEnable()
        {
            


        }

    
    void OnDisable()
    {
        if (renderCamera != null)
            renderCamera.targetTexture = null;
            
        if (renderTexture != null)
            renderTexture.Release();
    }
    
    // Método auxiliar para converter RenderTexture para Texture2D
    private Texture2D toTexture2D(RenderTexture rTex)
    {
        Texture2D tex = new Texture2D(rTex.width, rTex.height, TextureFormat.RGB24, false);
        RenderTexture.active = rTex;
        tex.ReadPixels(new Rect(0, 0, rTex.width, rTex.height), 0, 0);
        tex.Apply();
        return tex;
    }
    
    // Método para acessar a render texture de outros scripts
    public RenderTexture GetRenderTexture()
    {
        return renderTexture;
    }

    
    void SetupCamera()
    {

        renderTexture = new RenderTexture(textureWidth, textureHeight, 24);
        renderTexture.antiAliasing = 2;
        renderTexture.filterMode = FilterMode.Bilinear;
        
      
        if (renderCamera == null)
            renderCamera = GetComponent<Camera>();
            
        renderCamera.targetTexture = renderTexture;
        
        //  UI Image para mostrar a render texture
        if (uiImage != null)
        {
            Sprite sprite = Sprite.Create(
                toTexture2D(renderTexture),
                new Rect(0, 0, renderTexture.width, renderTexture.height),
                new Vector2(0.5f, 0.5f)
            );
            uiImage.sprite = sprite;
        }
        
        //  recomendado para render textures)
        if (rawImage != null)
        {
            rawImage.texture = renderTexture;
        }

    
            
        StartCoroutine(SendCameraFrames());
    }
    
    void SetupNetwork()
    {
        sendClient = new UdpClient();
        receiveClient = new UdpClient(PORT_RECEIVE);
    }
    
    void StartReceiving()
    {
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.Start();
    }


        IEnumerator SendCameraFrames()
        {
            Texture2D tex = new Texture2D(renderTexture.width, renderTexture.height, TextureFormat.RGB24, false);

            while (true)
            {
                try
                {
                    // Capturar frame da câmera
                    RenderTexture.active = renderTexture;
                    tex.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
                    tex.Apply();

                    // Converter para JPG
                    byte[] bytes = tex.EncodeToJPG(75);

                    // Enviar para o servidor
                    sendClient.Send(bytes, bytes.Length, serverIP, PORT_SEND);
                }
                catch (Exception e)
                {
                    Debug.LogWarning($"Erro ao enviar frame: {e.Message}");
                    speed=0;
                    steering=0;
                }

                yield return new WaitForSeconds(0.033f); // ~30 FPS
            }
        }


    
            
void ReceiveData()
{
    IPEndPoint remoteEndPoint = new IPEndPoint(IPAddress.Any, PORT_RECEIVE);

    while (isRunning)
    {
        try
        {
            byte[] receiveBytes = receiveClient.Receive(ref remoteEndPoint);
            if (receiveBytes.Length == 8) // 2 floats, 4 bytes cada
            {
                speed = BitConverter.ToSingle(receiveBytes, 0);
                steering = BitConverter.ToSingle(receiveBytes, 4);


            }
        }
        catch (SocketException e)
        {
            Debug.LogWarning($"Erro ao receber dados: {e.Message}");
            speed=0;
            steering=0;
            Thread.Sleep(10);
        }
        catch (Exception e)
        {
            Debug.LogError($"Erro inesperado ao receber dados: {e.Message}");
            speed=0;
            steering=0;
        }
    }
}



    
    void Update()
    {
        	//string text = $"Speed: {speed:F2}, Steering: {steering:F2}";

if (Input.GetKeyDown(KeyCode.R)) // Pressionar "R" para resetar
    {
        ResetCar();
    }


         if (Input.GetKeyDown(KeyCode.Escape))
            {
                Application.Quit(); // Fecha a aplicação
                #if UNITY_EDITOR
                UnityEditor.EditorApplication.isPlaying = false; // Para parar no editor
                #endif
            }

      if (target == null) return;



	      
    }

 void FixedUpdate()
        {
             if (target == null) return;

       

        float steerInput = Mathf.Clamp(steering, -1.0f, 1.0f);

	

		target.steerInput = steerInput;
		target.throttleInput =speed;// Mathf.Clamp(speed, -1.0f, 1.0f);;

       //         Debug.Log($"Speed: {speed}, Steering: {steering}");
     
       
        }

    
    void OnDestroy()
    {
        isRunning = false;
        if (receiveThread != null)
            receiveThread.Abort();
            
        if (sendClient != null)
            sendClient.Close();
            
        if (receiveClient != null)
            receiveClient.Close();
            
        if (renderTexture != null)
        {
            renderTexture.Release();
            Destroy(renderTexture);
        }
    }
void ResetCar()
{
    if (target == null) return;

    Rigidbody rb = target.GetComponent<Rigidbody>();

    // Restaurar posição e rotação
    target.transform.position = initialPosition;
    target.transform.rotation = initialRotation;
    speed=0;
    steering=0;

    if (rb != null)
    {
        rb.velocity = initialVelocity;
        rb.angularVelocity = initialAngularVelocity;
    }

    target.throttleInput = 0;
    target.steerInput = 0;

    Debug.Log("Carro resetado para a posição inicial.");
}

}
}
