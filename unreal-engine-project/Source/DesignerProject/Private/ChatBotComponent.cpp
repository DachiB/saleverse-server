#include "ChatBotComponent.h"
#include "Modules/ModuleManager.h"
#include "WebSocketsModule.h"
#include "IWebSocket.h"
#include "Async/Async.h"

DEFINE_LOG_CATEGORY(LogChatBot);

UChatBotComponent::UChatBotComponent()
{
    PrimaryComponentTick.bCanEverTick = false;
}

void UChatBotComponent::BeginPlay()
{
    Super::BeginPlay();
    if (bAutoConnect) Connect();
}

void UChatBotComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    Close();
    Super::EndPlay(EndPlayReason);
}

bool UChatBotComponent::IsConnected() const
{
    return Socket.IsValid() && Socket->IsConnected();
}

void UChatBotComponent::Connect()
{
    if (!FModuleManager::Get().IsModuleLoaded("WebSockets"))
    {
        FModuleManager::LoadModuleChecked<FWebSocketsModule>("WebSockets");
    }

    if (IsConnected())
    {
        UE_LOG(LogChatBot, Warning, TEXT("Already connected: %s"), *Url);
        return;
    }

    Socket = FWebSocketsModule::Get().CreateWebSocket(Url);
    Socket->OnConnected().AddUObject(this, &UChatBotComponent::HandleConnected);
    Socket->OnMessage().AddUObject(this, &UChatBotComponent::HandleMessage);
    Socket->OnClosed().AddUObject(this, &UChatBotComponent::HandleClosed);
    Socket->OnConnectionError().AddUObject(this, &UChatBotComponent::HandleError);

    UE_LOG(LogChatBot, Log, TEXT("Connecting to %s..."), *Url);
    Socket->Connect();
}

void UChatBotComponent::Close()
{
    if (Socket.IsValid())
    {
        UE_LOG(LogChatBot, Log, TEXT("Closing WebSocket"));
        Socket->Close();
    }
    ResetSocket();
}

void UChatBotComponent::ResetSocket()
{
    if (Socket.IsValid())
    {
        Socket->OnConnected().RemoveAll(this);
        Socket->OnMessage().RemoveAll(this);
        Socket->OnClosed().RemoveAll(this);
        Socket->OnConnectionError().RemoveAll(this);
        Socket.Reset();
    }
}

void UChatBotComponent::SendUserMessage(const FString& Text)
{
    if (!IsConnected())
    {
        UE_LOG(LogChatBot, Warning, TEXT("Not connected; cannot SendUserMessage"));
        return;
    }
    const FString Payload = FString::Printf(TEXT("USER|%s"), *Text);
    Socket->Send(Payload);
}

void UChatBotComponent::RequestSpecFromText(const FString& Text)
{
    if (!IsConnected())
    {
        UE_LOG(LogChatBot, Warning, TEXT("Not connected; cannot RequestSpecFromText"));
        return;
    }
    const FString Payload = FString::Printf(TEXT("SPEC|%s"), *Text);
    Socket->Send(Payload);
}

void UChatBotComponent::HandleConnected()
{
    UE_LOG(LogChatBot, Log, TEXT("Connected: %s"), *Url);
    AsyncTask(ENamedThreads::GameThread, [this]() { OnConnected.Broadcast(); });
}

void UChatBotComponent::HandleMessage(const FString& Msg)
{
    // Format expected: TYPE|payload  (TYPE in {CHUNK, FINAL, SPEC, MATSPEC, ERROR})
    const int32 PipeIdx = Msg.Find(TEXT("|"));
    FString Type = Msg;
    FString Payload;
    if (PipeIdx != INDEX_NONE)
    {
        Type = Msg.Left(PipeIdx);
        Payload = Msg.Mid(PipeIdx + 1);
    }

    AsyncTask(ENamedThreads::GameThread, [this, Type, Payload]()
        {
            if (Type.Equals(TEXT("CHUNK"), ESearchCase::IgnoreCase))
                OnChatChunk.Broadcast(Payload);
            else if (Type.Equals(TEXT("FINAL"), ESearchCase::IgnoreCase))
                OnChatFinal.Broadcast(Payload);
            else if (Type.Equals(TEXT("SPEC"), ESearchCase::IgnoreCase))
                OnSpecJson.Broadcast(Payload);
            else if (Type.Equals(TEXT("MATSPEC"), ESearchCase::IgnoreCase))   // NEW: material spec
                OnMatSpecJson.Broadcast(Payload);
            else if (Type.Equals(TEXT("ERROR"), ESearchCase::IgnoreCase))
                OnError.Broadcast(Payload);
            else
                UE_LOG(LogChatBot, Verbose, TEXT("Unhandled WS type: %s | %s"), *Type, *Payload);
        });
}

void UChatBotComponent::HandleClosed(int32 Status, const FString& Reason, bool bWasClean)
{
    UE_LOG(LogChatBot, Warning, TEXT("Socket closed (%d, clean=%s): %s"),
        Status, bWasClean ? TEXT("true") : TEXT("false"), *Reason);

    AsyncTask(ENamedThreads::GameThread, [this, Status, Reason]()
        {
            OnClosed.Broadcast(Status, Reason);
        });

    ResetSocket();
}

void UChatBotComponent::HandleError(const FString& Err)
{
    UE_LOG(LogChatBot, Error, TEXT("Socket error: %s"), *Err);
    AsyncTask(ENamedThreads::GameThread, [this, Err]()
        {
            OnError.Broadcast(Err);
        });
}
